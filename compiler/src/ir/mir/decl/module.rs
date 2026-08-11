//! Module lowering: registers type metadata and vtables, then lowers function
//! bodies in two passes so forward references resolve.

use super::*;
use crate::ir::drop_analysis::{DropBehavior, DropPointAnalyzer, DropPoints};
use crate::ir::hir::*;
use crate::ir::{
    BinaryOp, CallingConvention, CompareOp, EnvironmentLayout, FunctionKind,
    FunctionSignatureBuilder, IrBasicBlock, IrBlockId, IrBuilder, IrEnumVariant, IrField,
    IrFunction, IrFunctionId, IrFunctionSignature, IrGlobal, IrGlobalId, IrId, IrInstruction,
    IrLocal, IrModule, IrParameter, IrPhiNode, IrSourceLocation, IrTerminator, IrType, IrTypeDef,
    IrTypeDefId, IrTypeDefinition, IrValue, Linkage, UnaryOp,
};
use crate::stdlib::{IrTypeDescriptor, MethodSignature, StdlibMapping};
use crate::tast::symbols::SymbolFlags;
use crate::tast::{
    InternedString, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeKind,
    TypeTable,
};
use log::{debug, trace, warn};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

impl<'a> HirToMirContext<'a> {
    pub fn lower_module(&mut self, hir_module: &HirModule) -> Result<IrModule, Vec<LoweringError>> {
        // Extract SSA optimization hints from HIR metadata
        // These were populated during HIR lowering by querying DFG/SSA
        self.extract_ssa_hints_from_hir(hir_module);

        // Set module metadata
        self.builder.module.metadata.language_version =
            hir_module.metadata.language_version.clone();

        // IMPORTANT: Register type metadata FIRST before lowering any functions
        // This populates field_index_map which is needed for field access
        // Two-pass: register interfaces first so interface_method_names is populated
        // before classes try to build vtables.
        // Topological sort: parent interfaces must be registered before children
        // so inherited methods are available when building child method tables.
        {
            let interfaces: Vec<(TypeId, &HirTypeDecl)> = hir_module
                .types
                .iter()
                .filter(|(_, td)| matches!(td, HirTypeDecl::Interface(_)))
                .map(|(tid, td)| (*tid, td))
                .collect();

            let mut registered: std::collections::BTreeSet<TypeId> =
                std::collections::BTreeSet::new();
            let mut remaining = interfaces.clone();
            let max_iterations = remaining.len() + 1;
            let mut iteration = 0;
            while !remaining.is_empty() && iteration < max_iterations {
                iteration += 1;
                let mut next_remaining = Vec::new();
                for (type_id, type_decl) in remaining {
                    if let HirTypeDecl::Interface(iface) = type_decl {
                        let parents_ready = iface
                            .extends
                            .iter()
                            .all(|parent_tid| registered.contains(parent_tid));
                        if parents_ready {
                            self.register_type_metadata(type_id, type_decl);
                            registered.insert(type_id);
                        } else {
                            next_remaining.push((type_id, type_decl));
                        }
                    }
                }
                remaining = next_remaining;
            }
            // Register any remaining interfaces (handles cycles gracefully)
            for (type_id, type_decl) in remaining {
                self.register_type_metadata(type_id, type_decl);
            }
        }
        for (type_id, type_decl) in &hir_module.types {
            if !matches!(type_decl, HirTypeDecl::Interface(_)) {
                self.register_type_metadata(*type_id, type_decl);
            }
        }

        // Build class vtables after all type metadata is registered
        self.build_class_vtables();

        // CRITICAL: Two-pass lowering to handle forward references
        // Class methods might be lowered after module functions that call them,
        // causing "function not found" errors.
        //
        // Pass 1: Register ALL function signatures WITHOUT lowering bodies
        // This ensures function_map is fully populated before any calls are made

        // Pass 1a: Register class method signatures
        // eprintln!("[PASS1] {} types in HIR module '{}'", hir_module.types.len(), hir_module.name);
        for (type_id, type_decl) in &hir_module.types {
            match type_decl {
                HirTypeDecl::Class(class) => {
                    let cname = self.string_interner.get(class.name).unwrap_or("?");
                    // eprintln!("[PASS1] class={} ctor={} methods={}", cname, class.constructor.is_some(), class.methods.len());
                    // eprintln!(
                    //     "DEBUG Pass1a: Registering methods for class {:?}",
                    //     self.string_interner.get(class.name).unwrap_or("<unknown>")
                    // );
                    self.current_class_symbol = Some(class.symbol_id);
                    for method in &class.methods {
                        // Skip bodyless extern class methods. The stdlib mapping in Phase 2
                        // creates properly qualified functions (e.g., "haxe_bytes_get") via
                        // get_or_register_extern_function. Phase 1 stubs use bare method
                        // names (e.g., "get") which cause WASM import collisions.
                        if class.is_extern && method.function.body.is_none() {
                            continue;
                        }
                        let this_type = if !method.is_static {
                            Some(*type_id)
                        } else {
                            None
                        };
                        // Pass class type params for generic class methods
                        self.register_function_signature_with_class_type_params(
                            method.function.symbol_id,
                            &method.function,
                            this_type,
                            &class.type_params,
                        );
                    }

                    // Register constructor signature with class type params
                    if let Some(constructor) = &class.constructor {
                        if !class.is_extern {
                            self.register_constructor_signature_with_class_type_params(
                                class.symbol_id,
                                constructor,
                                *type_id,
                                &class.type_params,
                            );
                        }
                    }
                }
                HirTypeDecl::Abstract(abstract_decl) => {
                    // Register abstract method signatures — same as classes but
                    // this_type uses the underlying type (value, not pointer)
                    for method in &abstract_decl.methods {
                        let this_type = if !method.is_static {
                            Some(abstract_decl.underlying)
                        } else {
                            None
                        };
                        self.register_function_signature_with_class_type_params(
                            method.function.symbol_id,
                            &method.function,
                            this_type,
                            &abstract_decl.type_params,
                        );
                    }
                }
                _ => {}
            }
        }

        // Pass 1b: Register module function signatures
        for (symbol_id, hir_func) in &hir_module.functions {
            self.register_function_signature(*symbol_id, hir_func, None);
        }

        // Build reverse index for O(1) field-by-type lookups during expression lowering.
        // field_index_map is fully populated after class registration (Pass 1).
        self.rebuild_fields_by_type_cache();

        // Pass 2: Now lower all function bodies (both class methods and module functions)
        // At this point, function_map is fully populated

        // Pass 2a: Lower class methods and constructors
        for (type_id, type_decl) in &hir_module.types {
            let name_str = if let HirTypeDecl::Class(c) = type_decl {
                let n = self.string_interner.get(c.name).unwrap_or("<unknown>");
                n
            } else {
                "<not-a-class>"
            };
            // debug!("Pass2a: Processing type - TypeId={:?}, name={:?}", type_id, name_str);
            match type_decl {
                HirTypeDecl::Class(class) => {
                    // Get qualified class name for runtime mapping checks
                    // Prefer lowered @:native name (e.g., "rayzor::concurrent::Arc" -> "rayzor_concurrent_Arc")
                    // Fall back to qualified_name with underscores (e.g., "rayzor.Bytes" -> "rayzor_Bytes")
                    let qualified_class_name = self
                        .symbol_table
                        .get_symbol(class.symbol_id)
                        .and_then(|sym| {
                            // Prefer native name if available
                            if let Some(native) = sym.native_name {
                                self.string_interner
                                    .get(native)
                                    .map(|n| n.replace("::", "_"))
                            } else {
                                sym.qualified_name
                                    .and_then(|qn| self.string_interner.get(qn))
                                    .map(|qn| qn.replace(".", "_"))
                            }
                        });

                    // Fallback to simple class name if no qualified name
                    let class_name = self.string_interner.get(class.name);

                    // Lower each method body
                    for method in &class.methods {
                        // SPECIAL CASE: Skip lowering method if this is an extern class
                        // with a runtime mapping for this method. For extern classes like FileSystem,
                        // methods are handled by the runtime mapping system, not by MIR stubs.
                        let should_skip_method = if method.function.body.is_none() {
                            // Method has no body - check if it has a runtime mapping
                            // Try qualified name first (e.g., "rayzor_Bytes"), then fall back to simple name
                            let has_mapping = if let Some(method_name) =
                                self.string_interner.get(method.function.name)
                            {
                                // Try qualified name first
                                let found_in_qualified = qualified_class_name
                                    .as_ref()
                                    .map(|qn| {
                                        self.stdlib_mapping.has_mapping(
                                            qn,
                                            method_name,
                                            method.is_static,
                                        )
                                    })
                                    .unwrap_or(false);

                                if found_in_qualified {
                                    true
                                } else if let Some(class_name_str) = class_name {
                                    // Fall back to simple class name (e.g., "FileSystem" instead of "sys_FileSystem")
                                    self.stdlib_mapping.has_mapping(
                                        class_name_str,
                                        method_name,
                                        method.is_static,
                                    )
                                } else {
                                    false
                                }
                            } else {
                                false
                            };

                            // Also skip if this is an extern class - extern classes have
                            // their methods handled by runtime mappings or MIR wrappers
                            let is_extern_class = self
                                .symbol_table
                                .get_symbol(class.symbol_id)
                                .map(|sym| {
                                    sym.flags
                                        .contains(crate::tast::symbols::SymbolFlags::EXTERN)
                                })
                                .unwrap_or(false);

                            has_mapping || is_extern_class
                        } else {
                            false
                        };

                        if should_skip_method {
                            continue;
                        }

                        // Skip synthetic methods on @:cstruct / @:gpuStruct — handled at call site
                        {
                            let has_no_body = method.function.body.is_none()
                                || method
                                    .function
                                    .body
                                    .as_ref()
                                    .map(|b| b.statements.is_empty())
                                    .unwrap_or(false);
                            if has_no_body {
                                if let Some(method_name) =
                                    self.string_interner.get(method.function.name)
                                {
                                    if method_name == "cdef" {
                                        let is_cstruct = self
                                            .symbol_table
                                            .get_symbol(class.symbol_id)
                                            .map(|sym| sym.flags.is_cstruct())
                                            .unwrap_or(false);
                                        if is_cstruct {
                                            self.get_or_compute_cstruct_layout(*type_id);
                                            continue;
                                        }
                                    }
                                    if matches!(method_name, "gpuDef" | "gpuSize" | "gpuAlignment")
                                    {
                                        let is_gpu_struct = self
                                            .symbol_table
                                            .get_symbol(class.symbol_id)
                                            .map(|sym| sym.flags.is_gpu_struct())
                                            .unwrap_or(false);
                                        if is_gpu_struct {
                                            self.get_or_compute_gpu_struct_layout(*type_id);
                                            continue;
                                        }
                                    }
                                }
                            }
                        }

                        if method.is_static {
                            self.lower_function_body(
                                method.function.symbol_id,
                                &method.function,
                                None,
                                Some(class.symbol_id),
                            );
                        } else {
                            self.lower_function_body(
                                method.function.symbol_id,
                                &method.function,
                                Some(*type_id),
                                Some(class.symbol_id),
                            );
                        }
                    }

                    // Lower constructor body
                    if let Some(constructor) = &class.constructor {
                        // SPECIAL CASE: Skip lowering constructor if this is an extern class
                        // with a runtime mapping for "new". For extern classes like Channel, Thread, etc.,
                        // the "new" constructor is handled by the runtime mapping system, not by
                        // generating a MIR constructor function.
                        // Try qualified class name first (e.g., "rayzor_Bytes"), then fall back to simple name
                        let should_skip_constructor = {
                            // Helper to check runtime mapping for a class name
                            let check_class_runtime = |name: &str| -> bool {
                                if let Some(class_name_static) =
                                    self.stdlib_mapping.get_class_static_str(name)
                                {
                                    let method_sig =
                                        crate::stdlib::runtime_mapping::MethodSignature {
                                            class: class_name_static,
                                            method: "new",
                                            is_static: true,
                                            is_constructor: true,
                                            param_count: 0,
                                        };
                                    if self.stdlib_mapping.get(&method_sig).is_some() {
                                        debug!(
                                            "Skipping constructor lowering for extern class '{}' - using runtime mapping",
                                            name
                                        );
                                        return true;
                                    }
                                }
                                false
                            };

                            // Try qualified name first, then fall back to simple name
                            let found_in_qualified = qualified_class_name
                                .as_ref()
                                .map(|qn| check_class_runtime(qn))
                                .unwrap_or(false);

                            if found_in_qualified {
                                true
                            } else if let Some(class_name_str) = class_name {
                                // Fallback to simple class name (e.g., "Thread" instead of "rayzor_concurrent_Thread")
                                if check_class_runtime(class_name_str) {
                                    true
                                } else {
                                    // Also skip if this is an extern class (using TAST flags)
                                    self.symbol_table
                                        .get_symbol(class.symbol_id)
                                        .map(|sym| {
                                            sym.flags
                                                .contains(crate::tast::symbols::SymbolFlags::EXTERN)
                                        })
                                        .unwrap_or(false)
                                }
                            } else {
                                false
                            }
                        };

                        if !should_skip_constructor {
                            // debug!("Lowering constructor for class {:?}", class.name);
                            self.lower_constructor_body(
                                class.symbol_id,
                                constructor,
                                *type_id,
                                class.extends,
                            );
                        }
                    }
                }
                HirTypeDecl::Abstract(abstract_decl) => {
                    // Lower abstract method bodies — same as classes but
                    // this_type uses the underlying type (value, not pointer)
                    for method in &abstract_decl.methods {
                        if method.function.body.is_none() {
                            continue; // Skip extern abstract methods
                        }
                        let this_type = if !method.is_static {
                            Some(abstract_decl.underlying)
                        } else {
                            None
                        };
                        self.lower_function_body(
                            method.function.symbol_id,
                            &method.function,
                            this_type,
                            Some(abstract_decl.symbol_id),
                        );
                    }
                }
                _ => {}
            }
        }

        // Pass 2b: Lower module function bodies
        for (symbol_id, hir_func) in &hir_module.functions {
            self.lower_function_body(*symbol_id, hir_func, None, None);
        }

        // Lower globals
        // Sort globals by symbol ID for deterministic lowering order.
        // globals is a BTreeMap — unsorted iteration causes non-deterministic
        // function ID assignment when global initializers create extern functions.
        let mut sorted_globals: Vec<_> = hir_module.globals.iter().collect();
        sorted_globals.sort_by_key(|(sid, _)| sid.as_raw());
        for (symbol_id, global) in sorted_globals {
            self.lower_global(*symbol_id, global);
        }

        // Generate reflective constructor wrappers for Type.createInstance().
        self.generate_constructor_reflect_wrappers();

        // Generate __vtable_init__ function for class virtual dispatch tables
        if !self.class_vtables.is_empty() || !self.constructor_reflect_wrappers.is_empty() {
            self.generate_vtable_init_function();
        }

        // Generate __init__ whenever the module has globals so repeated executions
        // can restore static state even if every initializer is constant/defaulted.
        if !self.dynamic_globals.is_empty() || !self.builder.module.globals.is_empty() {
            self.generate_module_init_function();
        }

        if self.errors.is_empty() {
            // eprintln!(
            //     "  ℹ️  Returning MIR module with {} functions, {} extern_functions",
            //     self.builder.module.functions.len(),
            //     self.builder.module.extern_functions.len()
            // );

            Ok(std::mem::replace(
                &mut self.builder.module,
                IrModule::new(String::new(), String::new()),
            ))
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }
}
