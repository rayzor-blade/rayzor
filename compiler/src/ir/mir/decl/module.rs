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
        // Hints were populated during HIR lowering by querying DFG/SSA.
        self.extract_ssa_hints_from_hir(hir_module);

        self.builder.module.metadata.language_version =
            hir_module.metadata.language_version.clone();

        // Type metadata must be registered before any function is lowered: it
        // populates field_index_map, which field access needs. Interfaces come
        // before classes so interface_method_names exists when classes build
        // vtables, and parent interfaces before children so inherited methods
        // are available for the child method tables.
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

        // Two-pass lowering handles forward references: a class method may be
        // lowered after a module function that calls it.
        //
        // Pass 1 registers ALL function signatures without lowering bodies, so
        // function_map is complete before any call is emitted.

        // Pass 1a: class method signatures
        for (type_id, type_decl) in &hir_module.types {
            match type_decl {
                HirTypeDecl::Class(class) => {
                    let cname = self.string_interner.get(class.name).unwrap_or("?");
                    self.current_class_symbol = Some(class.symbol_id);
                    for method in &class.methods {
                        // Skip bodyless extern class methods: the stdlib mapping
                        // registers them under qualified names (e.g.
                        // "haxe_bytes_get"), while a Pass 1 stub would use the
                        // bare name and collide on WASM imports.
                        if class.is_extern && method.function.body.is_none() {
                            continue;
                        }
                        let this_type = if !method.is_static {
                            Some(*type_id)
                        } else {
                            None
                        };
                        self.register_function_signature_with_class_type_params(
                            method.function.symbol_id,
                            &method.function,
                            this_type,
                            &class.type_params,
                        );
                    }

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

        // Pass 2 lowers function bodies; function_map is complete by now.

        // Pass 2a: class methods and constructors
        for (type_id, type_decl) in &hir_module.types {
            let name_str = if let HirTypeDecl::Class(c) = type_decl {
                let n = self.string_interner.get(c.name).unwrap_or("<unknown>");
                n
            } else {
                "<not-a-class>"
            };
            match type_decl {
                HirTypeDecl::Class(class) => {
                    // The class's registered mapping key, resolved from the
                    // strongest name it carries: @:native, then the qualified
                    // name, then the simple name.
                    let qualified_class_name = self
                        .symbol_table
                        .get_symbol(class.symbol_id)
                        .and_then(|sym| {
                            if let Some(native) = sym.native_name {
                                self.string_interner
                                    .get(native)
                                    .map(|n| self.canonical_class_spelling(n))
                            } else {
                                sym.qualified_name
                                    .and_then(|qn| self.string_interner.get(qn))
                                    .map(|qn| self.canonical_class_spelling(qn))
                            }
                        })
                        .or_else(|| {
                            self.string_interner
                                .get(class.name)
                                .and_then(|n| self.stdlib_mapping.get_class_static_str(n))
                                .map(str::to_string)
                        });

                    for method in &class.methods {
                        // A method of an extern class with a runtime mapping is
                        // served by the mapping system, not by a MIR stub.
                        let should_skip_method = if method.function.body.is_none() {
                            let has_mapping = if let Some(method_name) =
                                self.string_interner.get(method.function.name)
                            {
                                qualified_class_name
                                    .as_ref()
                                    .and_then(|qn| self.stdlib_mapping.class_key(qn))
                                    .map(|key| {
                                        self.stdlib_mapping.has_mapping(
                                            key,
                                            method_name,
                                            method.is_static,
                                        )
                                    })
                                    .unwrap_or(false)
                            } else {
                                false
                            };

                            // Extern classes get their methods from runtime
                            // mappings or MIR wrappers either way.
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

                        // An abstract method contributes its signature and its
                        // vtable slot, which the declare pass has already
                        // registered; the body belongs to whichever subclass
                        // overrides it.
                        if method.is_abstract {
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

                    if let Some(constructor) = &class.constructor {
                        // An extern class with a runtime mapping for "new" gets
                        // its constructor from the mapping system, so no MIR
                        // constructor is generated. Qualified class name first
                        // (e.g. "rayzor_Bytes"), then the simple name.
                        let should_skip_constructor = {
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

                            let found_in_qualified = qualified_class_name
                                .as_ref()
                                .map(|qn| check_class_runtime(qn))
                                .unwrap_or(false);

                            if found_in_qualified {
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
                        };

                        if !should_skip_constructor {
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

        // Sort globals by symbol id: unsorted iteration makes function-id
        // assignment non-deterministic when an initializer registers an extern
        // function.
        let mut sorted_globals: Vec<_> = hir_module.globals.iter().collect();
        sorted_globals.sort_by_key(|(sid, _)| sid.as_raw());
        for (symbol_id, global) in sorted_globals {
            self.lower_global(*symbol_id, global);
        }

        // Generate reflective constructor wrappers for Type.createInstance().
        self.generate_constructor_reflect_wrappers();

        if !self.class_vtables.is_empty() || !self.constructor_reflect_wrappers.is_empty() {
            self.generate_vtable_init_function();
        }

        // Generate __init__ whenever the module has globals so repeated executions
        // can restore static state even if every initializer is constant/defaulted.
        if !self.dynamic_globals.is_empty() || !self.builder.module.globals.is_empty() {
            self.generate_module_init_function();
        }

        if self.errors.is_empty() {
            Ok(std::mem::replace(
                &mut self.builder.module,
                IrModule::new(String::new(), String::new()),
            ))
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }
}
