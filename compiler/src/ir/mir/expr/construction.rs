//! `new`: allocation, field initialisation, and the constructor call.

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
    pub(crate) fn lower_new(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::New {
            class_type,
            type_args: hir_type_args,
            args,
            class_name: hir_class_name,
            ..
        } = &expr.kind
        else {
            unreachable!("lower_new on a non-New expression")
        };
        let debug_class_name =
            hir_class_name.and_then(|interned| self.string_interner.get(interned));
        debug!(
            "[NEW EXPR]: class_type={:?}, args.len={}, hir_class_name={:?}, hir_type_args={:?}",
            class_type,
            args.len(),
            debug_class_name,
            hir_type_args
        );

        // Check if this is an abstract type
        let type_table = self.type_table;
        let (is_abstract, actual_symbol_id) = if let Some(type_ref) = type_table.get(*class_type) {
            let symbol_id = match &type_ref.kind {
                crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                crate::tast::TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                crate::tast::TypeKind::GenericInstance { base_type, .. } => {
                    // Unwrap GenericInstance to find base class symbol_id
                    if let Some(base_info) = type_table.get(*base_type) {
                        match &base_info.kind {
                            crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                crate::tast::TypeKind::TypeAlias {
                    symbol_id,
                    target_type,
                    ..
                } => {
                    // Resolve TypeAlias to find the actual class symbol
                    // For user classes, the TypeAlias symbol_id IS the class symbol
                    let mut resolved = Some(*symbol_id);
                    // Also try resolving through target_type for deeper aliases
                    if let Some(target) = type_table.get(*target_type) {
                        if let crate::tast::TypeKind::Class {
                            symbol_id: class_sym,
                            ..
                        } = &target.kind
                        {
                            resolved = Some(*class_sym);
                        }
                    }
                    resolved
                }
                _ => None,
            };
            let is_abs = matches!(type_ref.kind, crate::tast::TypeKind::Abstract { .. });
            (is_abs, symbol_id)
        } else {
            (false, None)
        };

        // Cross-module `new C()` where C is an imported user class often
        // arrives with `class_type` as a Placeholder (the class metadata
        // wasn't resolved in THIS lowering context), so `actual_symbol_id`
        // is None. That degrades the allocation to a bare 16-byte block
        // with a ZERO class-id header, NO constructor call, and NO
        // interface fat-ptr wrap — the exact shape that made
        // `ArchRegistry.withDefaults`'s `new LlamaArch()` produce a
        // headerless object whose later interface dispatch read garbage
        // (Void-tagged receiver → SIGSEGV). Recover the class SymbolId by
        // name (Placeholder name or the HIR class name), mirroring the
        // interface machinery's name-based lazy resolution. This restores
        // the proper alloc size, header id, ctor call, and iface wrap.
        let actual_symbol_id = actual_symbol_id.or_else(|| {
            let placeholder_name = type_table.get(*class_type).and_then(|t| {
                if let crate::tast::TypeKind::Placeholder { name } = &t.kind {
                    self.string_interner.get(*name)
                } else {
                    None
                }
            });
            let by_name = placeholder_name
                .or_else(|| hir_class_name.and_then(|interned| self.string_interner.get(interned)));
            by_name.and_then(|name| self.lookup_class_symbol_by_name(name))
        });

        // SPECIAL CASE: Abstract type constructors
        // If this is an abstract type, treat this as a simple value wrap (no allocation).
        if is_abstract {
            // Before treating as abstract value-wrap, check if there's a user-defined
            // class with the same name that has a constructor registered. User classes
            // should override stdlib abstract types with the same short name
            // (e.g., user's `class Box<T>` vs stdlib `rayzor.Box<T>` abstract).
            let has_user_constructor = hir_class_name
                .and_then(|interned| self.string_interner.get(interned))
                .map(|name| self.constructor_name_map.contains_key(name))
                .unwrap_or(false)
                || actual_symbol_id
                    .and_then(|sid| self.symbol_table.get_symbol(sid))
                    .and_then(|s| s.qualified_name)
                    .and_then(|qn| self.string_interner.get(qn))
                    .map(|qn| self.constructor_name_map.contains_key(qn))
                    .unwrap_or(false);

            if !has_user_constructor {
                if args.len() == 1 {
                    return self.lower_expression(&args[0]);
                } else if args.is_empty() {
                    return self.builder.build_const(IrValue::I32(0));
                } else {
                    return self.lower_expression(&args[0]);
                }
            }
            // User constructor exists - fall through to normal class construction
        }

        // SPECIAL CASE: Check if this is an extern stdlib class constructor BEFORE fallback
        // For extern stdlib classes (Channel, Thread, Arc, Mutex), we call the MIR wrapper
        // function (e.g., Channel_init) instead of allocating and calling a constructor.
        // This MUST come before the value-wrap fallback to prevent returning the argument value!

        // PRIORITY #1: Use class_name from HIR if available (preserves actual class name even when TypeId is invalid)
        let mut class_name = hir_class_name.and_then(|interned| self.string_interner.get(interned));

        // FALLBACK #1: Try to get class name from TypeId if HIR didn't have it
        if class_name.is_none() {
            let type_table = self.type_table;
            class_name = if let Some(type_ref) = type_table.get(*class_type) {
                match &type_ref.kind {
                    crate::tast::TypeKind::Class { symbol_id, .. } => self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|sym| self.string_interner.get(sym.name)),
                    crate::tast::TypeKind::GenericInstance { base_type, .. } => {
                        // Unwrap GenericInstance to get base class name
                        if let Some(base_info) = type_table.get(*base_type) {
                            if let crate::tast::TypeKind::Class { symbol_id, .. } = &base_info.kind
                            {
                                self.symbol_table
                                    .get_symbol(*symbol_id)
                                    .and_then(|sym| self.string_interner.get(sym.name))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };
        }

        // FALLBACK #2: If TypeId lookup failed (e.g., for extern stdlib classes that aren't
        // pre-registered because Channel.hx is skipped), try getting class name from the
        // actual_symbol_id which comes from the HIR New expression
        if class_name.is_none() {
            if let Some(sym_id) = actual_symbol_id {
                class_name = self
                    .symbol_table
                    .get_symbol(sym_id)
                    .and_then(|sym| self.string_interner.get(sym.name));
            }
        }

        // FALLBACK #3: If still no class name and TypeId is invalid (u32::MAX),
        // try checking all stdlib registered class names to see if ANY constructor matches
        // This is a last resort for extern stdlib classes that weren't pre-registered
        if class_name.is_none() && *class_type == TypeId::from_raw(u32::MAX) {
            // Get ALL classes that have registered constructors from the stdlib mapping
            let constructor_classes = self.stdlib_mapping.get_constructor_classes();

            // Try each registered constructor class
            for potential_class in constructor_classes {
                let method_sig = crate::stdlib::runtime_mapping::MethodSignature {
                    class: potential_class,
                    method: "new",
                    is_static: true,
                    is_constructor: true,
                    param_count: 0,
                };
                if self.stdlib_mapping.get(&method_sig).is_some() {
                    class_name = Some(potential_class);
                    break;
                }
            }
        }

        // MONOMORPHIZATION: For generic extern classes like Vec<T>, monomorphize the class name
        // based on type arguments. Vec<Int> -> VecI32, Vec<Float> -> VecF64, etc.
        // Use hir_type_args directly (from HIR) instead of type_table lookup (which may fail for extern classes)
        let monomorphized_class_name: Option<String> = if let Some(base_name) = class_name {
            if base_name == "Vec" && !hir_type_args.is_empty() {
                // Get the first type argument and determine the monomorphized suffix
                let first_arg = hir_type_args[0];
                let type_table = self.type_table;
                let suffix = if let Some(arg_type) = type_table.get(first_arg) {
                    match &arg_type.kind {
                        crate::tast::TypeKind::Int => Some("I32"),
                        crate::tast::TypeKind::Float => Some("F64"),
                        crate::tast::TypeKind::Bool => Some("Bool"),
                        crate::tast::TypeKind::String => Some("Ptr"),
                        crate::tast::TypeKind::Class { symbol_id, .. } => {
                            // Check if it's Int64 (a class type representing 64-bit int)
                            if let Some(class_info) = self.symbol_table.get_symbol(*symbol_id) {
                                if let Some(name) = self.string_interner.get(class_info.name) {
                                    if name == "Int64" {
                                        Some("I64")
                                    } else {
                                        Some("Ptr") // Other classes are reference types
                                    }
                                } else {
                                    Some("Ptr")
                                }
                            } else {
                                Some("Ptr")
                            }
                        }
                        _ => Some("Ptr"),
                    }
                } else {
                    Some("Ptr") // If type not found, default to Ptr variant
                };
                suffix.map(|s| format!("Vec{}", s))
            } else {
                None
            }
        } else {
            None
        };

        // Use monomorphized name if available, otherwise use original class name
        let final_class_name = monomorphized_class_name.as_deref().or(class_name);
        debug!(
            "[NEW EXPR]: final_class_name={:?} (monomorphized from {:?})",
            final_class_name, class_name
        );

        if let Some(class_name) = final_class_name {
            // Check if this class has a "new" constructor registered in the runtime mapping
            // Use find_constructor to look up the registered constructor mapping
            // This returns both the MethodSignature and RuntimeFunctionCall from the registry
            // PRIORITY: Try HIR class name first (may be fully qualified, e.g., "sys.ssl.Socket")
            // This is critical for subclasses like sys.ssl.Socket that extend sys.net.Socket,
            // because the symbol's qualified_name resolves to the parent class.
            // Look up constructor: extract needed data (Copy types) to avoid holding
            // a borrow on self.stdlib_mapping while calling self.lower_expression later.
            let arg_count = args.len();
            let constructor_info: Option<(&'static str, bool, bool)> = {
                let mut found = None;

                // Helper: try constructor lookup with param count first, then without.
                // This ensures overloaded constructors (e.g., Uncompress with 0 or 1 args)
                // select the correct overload.
                macro_rules! try_find_ctor {
                    ($name:expr) => {
                        if found.is_none() {
                            // Try param-count-aware lookup first
                            if let Some((_, rc)) = self
                                .stdlib_mapping
                                .find_constructor_with_params($name, arg_count)
                            {
                                found =
                                    Some((rc.runtime_name, rc.needs_out_param, rc.is_mir_wrapper));
                            }
                            // Fall back to any-param lookup
                            else if let Some((_, rc)) =
                                self.stdlib_mapping.find_constructor($name)
                            {
                                found =
                                    Some((rc.runtime_name, rc.needs_out_param, rc.is_mir_wrapper));
                            }
                        }
                    };
                }

                // PRIORITY #0: Try HIR class name as qualified (e.g., "sys.ssl.Socket" -> "sys_ssl_Socket")
                // This handles subclass constructors before the symbol-based lookup (which may
                // resolve to the parent class due to class inheritance).
                if class_name.contains('.') {
                    let qualified_hir = class_name.replace(".", "_");
                    try_find_ctor!(&qualified_hir);
                }
                if found.is_none() {
                    if let Some(sym_id) = actual_symbol_id {
                        if let Some(sym) = self.symbol_table.get_symbol(sym_id) {
                            // Try lowered @:native name first
                            if let Some(native) = sym.native_name {
                                if let Some(native_str) = self.string_interner.get(native) {
                                    let native_class_name = native_str.replace("::", "_");
                                    try_find_ctor!(&native_class_name);
                                }
                            }
                            // Fall back to qualified name
                            if found.is_none() {
                                if let Some(qn) = sym.qualified_name {
                                    if let Some(qual_name) = self.string_interner.get(qn) {
                                        let qualified_class_name = qual_name.replace(".", "_");
                                        try_find_ctor!(&qualified_class_name);
                                    }
                                }
                            }
                        }
                    }
                    // FALLBACK: Try simple class name
                    try_find_ctor!(class_name);

                    // FALLBACK #2: Try bare class name for qualified paths
                    // (e.g., "haxe.ds.ObjectMap" -> "ObjectMap")
                    if found.is_none() && class_name.contains('.') {
                        let bare_name = class_name.rsplit('.').next().unwrap_or(class_name);
                        try_find_ctor!(bare_name);
                    }
                } // close PRIORITY #0 if found.is_none()
                found
            };
            if let Some((wrapper_name, needs_out_param, is_mir_wrapper)) = constructor_info {
                // Lower arguments
                let arg_regs: Vec<_> = args
                    .iter()
                    .filter_map(|a| self.lower_expression(a))
                    .collect();

                // Register forward ref if not already present
                let param_types: Vec<IrType> = arg_regs
                    .iter()
                    .map(|reg| self.builder.get_register_type(*reg).unwrap_or(IrType::Any))
                    .collect();

                // For extern classes, the return type should be a pointer (opaque handle),
                // not the class struct itself
                let result_type = IrType::Ptr(Box::new(IrType::U8));

                // MIR wrappers are compiled by Cranelift alongside user code -> use forward ref
                // needs_out_param constructors also use MIR forward refs (legacy path)
                // Direct extern calls are for non-wrapper, non-out-param constructors
                let wrapper_func_id = if is_mir_wrapper || needs_out_param {
                    self.register_stdlib_mir_forward_ref(
                        wrapper_name,
                        param_types,
                        result_type.clone(),
                    )
                } else {
                    // Simple constructors are direct extern calls
                    self.get_or_register_extern_function(
                        wrapper_name,
                        param_types,
                        result_type.clone(),
                    )
                };

                // Call the wrapper and return the result
                let result = self
                    .builder
                    .build_call_direct(wrapper_func_id, arg_regs, result_type);
                if let Some(reg) = result {
                    self.register_class_hints
                        .insert(reg, class_name.to_string());
                }
                return result;
            }
        }

        // Check if constructor exists - try both TypeId and TypeId derived from SymbolId
        let mut constructor_type_id = *class_type;
        let mut has_constructor = self.constructor_map.contains_key(class_type);
        let mut ctor_path = if has_constructor { "typeid" } else { "none" };

        // If not found and we have a SymbolId, try TypeId derived from SymbolId as fallback.
        //
        // SymbolId and TypeId are DIFFERENT numbering spaces sharing this
        // map, so this lookup can land on another class's entry: `new
        // ChatResponse` picked up GenerationLoop's ctor (cached at
        // GenerationLoop's TypeId, numerically equal to ChatResponse's
        // SymbolId) and the Int passed in a String slot SIGSEGV'd at
        // runtime — host-dependent, since symbol numbering shifts with
        // module count. Accept the pun only when the candidate's owning
        // class matches the class being constructed; an unowned entry
        // passes (bridge for ctors registered before owners existed).
        if !has_constructor {
            if let Some(sym_id) = actual_symbol_id {
                let type_id_from_symbol = TypeId::from_raw(sym_id.as_raw());
                if let Some(&cand) = self.constructor_map.get(&type_id_from_symbol) {
                    let owner_ok = match (self.constructor_owner_map.get(&cand), debug_class_name) {
                        (Some(owner), Some(target)) => Self::class_names_match(owner, target),
                        _ => true,
                    };
                    if owner_ok {
                        constructor_type_id = type_id_from_symbol;
                        has_constructor = true;
                        ctor_path = "SYMBOL-PUN";
                    } else if std::env::var("RAYZOR_CTOR_DEBUG").is_ok() {
                        eprintln!(
                            "[CTOR] class={:?} REFUSED pun candidate {:?} (owner={:?})",
                            debug_class_name,
                            cand,
                            self.constructor_owner_map.get(&cand)
                        );
                    }
                }
            }
        }

        // If not found and this is a GenericInstance, try the base class TypeId
        if !has_constructor {
            let type_table = self.type_table;
            if let Some(type_info) = type_table.get(*class_type) {
                if let crate::tast::TypeKind::GenericInstance { base_type, .. } = &type_info.kind {
                    if self.constructor_map.contains_key(base_type) {
                        constructor_type_id = *base_type;
                        has_constructor = true;
                        ctor_path = "generic-base";
                    }
                }
            }
        }

        // Try constructor_name_map as final fallback before value wrap
        // Check bare name first, then qualified name from symbol table
        if !has_constructor {
            if let Some(class_name) = final_class_name {
                if let Some(&func_id) = self.constructor_name_map.get(class_name) {
                    constructor_type_id = *class_type;
                    has_constructor = true;
                    ctor_path = "bare-name";
                    self.constructor_map.insert(*class_type, func_id);
                    let bare = class_name.rsplit('.').next().unwrap_or(class_name);
                    self.constructor_owner_map
                        .entry(func_id)
                        .or_insert_with(|| bare.to_string());
                }
            }
            // Also try qualified name (e.g., "haxe.ds.List" vs bare "List")
            if !has_constructor {
                if let Some(sym_id) = actual_symbol_id {
                    if let Some(sym) = self.symbol_table.get_symbol(sym_id) {
                        if let Some(qn) = sym.qualified_name {
                            if let Some(qual_name) = self.string_interner.get(qn) {
                                if let Some(&func_id) = self.constructor_name_map.get(qual_name) {
                                    constructor_type_id = *class_type;
                                    has_constructor = true;
                                    ctor_path = "qual-name";
                                    self.constructor_map.insert(*class_type, func_id);
                                    let bare = qual_name.rsplit('.').next().unwrap_or(qual_name);
                                    self.constructor_owner_map
                                        .entry(func_id)
                                        .or_insert_with(|| bare.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        // If no constructor exists and we have exactly one argument, treat as value wrap
        // This handles abstract types that weren't properly detected above
        if !has_constructor && args.len() == 1 {
            let result = self.lower_expression(&args[0]);
            return result;
        }

        // SPECIAL CASE: Array constructor (@:coreType extern class)
        // Array needs special handling - call haxe_array_new() runtime function
        let type_table = self.type_table;
        let is_array = if let Some(type_ref) = type_table.get(*class_type) {
            matches!(type_ref.kind, crate::tast::TypeKind::Array { .. })
        } else {
            false
        };

        if is_array {
            // Allocate HaxeArray struct on heap (32 bytes = 4 x 8 for ptr, len, cap, elem_size)
            // Must be heap-allocated because array pointers can escape the creating function
            // (e.g., stored in class fields, returned from functions)
            let array_ptr = self.build_heap_alloc(32)?;

            // Zero-initialize the HaxeArray struct (ptr=null, len=0, cap=0, elem_size=8)
            // This represents an empty uninitialized array
            if let Some(zero_i64) = self.builder.build_const(IrValue::I64(0)) {
                // Zero out ptr field (offset 0)
                if let Some(index_0) = self.builder.build_const(IrValue::I32(0)) {
                    if let Some(ptr_field) =
                        self.builder
                            .build_gep(array_ptr, vec![index_0], IrType::I64)
                    {
                        self.builder.build_store(ptr_field, zero_i64);
                    }
                }
                // Zero out len field (offset 8)
                if let Some(index_1) = self.builder.build_const(IrValue::I32(1)) {
                    if let Some(len_field) =
                        self.builder
                            .build_gep(array_ptr, vec![index_1], IrType::I64)
                    {
                        self.builder.build_store(len_field, zero_i64);
                    }
                }
                // Zero out cap field (offset 16)
                if let Some(index_2) = self.builder.build_const(IrValue::I32(2)) {
                    if let Some(cap_field) =
                        self.builder
                            .build_gep(array_ptr, vec![index_2], IrType::I64)
                    {
                        self.builder.build_store(cap_field, zero_i64);
                    }
                }
                // Set elem_size field to 8 bytes (offset 24) - assume pointer size for now
                if let Some(elem_size_val) = self.builder.build_const(IrValue::I64(8)) {
                    if let Some(index_3) = self.builder.build_const(IrValue::I32(3)) {
                        if let Some(elem_size_field) =
                            self.builder
                                .build_gep(array_ptr, vec![index_3], IrType::I64)
                        {
                            self.builder.build_store(elem_size_field, elem_size_val);
                        }
                    }
                }
            }

            // Return the zero-initialized array pointer
            return Some(array_ptr);
        }

        // @:cstruct CLASS: flat C-compatible allocation (no object header)
        if self.is_cstruct_class(*class_type) {
            if let Some(layout) = self.get_or_compute_cstruct_layout(*class_type) {
                let obj_ptr = self.build_heap_alloc(layout.total_size as u64)?;

                // Zero-initialize all bytes via first field store of 0
                // (memset-style zero init would be better but this works)
                if let Some(zero) = self.builder.build_const(IrValue::I64(0)) {
                    for field in &layout.fields {
                        let offset_const = self
                            .builder
                            .build_const(IrValue::I64(field.byte_offset as i64))?;
                        let field_ptr = self.builder.build_ptr_add(
                            obj_ptr,
                            offset_const,
                            IrType::Ptr(Box::new(IrType::U8)),
                        )?;
                        let zero_val = match &field.ir_type {
                            IrType::F64 => self.builder.build_const(IrValue::F64(0.0))?,
                            IrType::I64 => self.builder.build_const(IrValue::I64(0))?,
                            IrType::I32 => self.builder.build_const(IrValue::I32(0))?,
                            IrType::Bool => self.builder.build_const(IrValue::Bool(false))?,
                            _ => self.builder.build_const(IrValue::I64(0))?,
                        };
                        self.builder.build_store(field_ptr, zero_val);
                    }
                }

                // Call constructor if exists
                let constructor_func_id = self.constructor_map.get(&constructor_type_id).copied();
                if let Some(constructor_func_id) = constructor_func_id {
                    let arg_regs: Vec<_> = std::iter::once(obj_ptr)
                        .chain(args.iter().filter_map(|a| self.lower_expression(a)))
                        .collect();
                    self.builder
                        .build_call_direct(constructor_func_id, arg_regs, IrType::Void);
                }

                return Some(obj_ptr);
            }
        }

        // @:gpuStruct CLASS: GPU-compatible flat allocation (no object header)
        // Uses 4-byte floats, 4-byte ints — matches Metal/CUDA struct layout
        if self.is_gpu_struct_class(*class_type) {
            if let Some(layout) = self.get_or_compute_gpu_struct_layout(*class_type) {
                let obj_ptr = self.build_heap_alloc(layout.total_size as u64)?;

                // Zero-initialize all fields
                if let Some(_zero) = self.builder.build_const(IrValue::I32(0)) {
                    for field in &layout.fields {
                        let offset_const = self
                            .builder
                            .build_const(IrValue::I64(field.byte_offset as i64))?;
                        let field_ptr = self.builder.build_ptr_add(
                            obj_ptr,
                            offset_const,
                            IrType::Ptr(Box::new(IrType::U8)),
                        )?;
                        let zero_val = match &field.ir_type {
                            IrType::F32 => self.builder.build_const(IrValue::F32(0.0))?,
                            IrType::I32 => self.builder.build_const(IrValue::I32(0))?,
                            _ => self.builder.build_const(IrValue::I32(0))?,
                        };
                        self.builder.build_store(field_ptr, zero_val);
                    }
                }

                // Call constructor if exists
                let constructor_func_id = self.constructor_map.get(&constructor_type_id).copied();
                if let Some(constructor_func_id) = constructor_func_id {
                    let arg_regs: Vec<_> = std::iter::once(obj_ptr)
                        .chain(args.iter().filter_map(|a| self.lower_expression(a)))
                        .collect();
                    self.builder
                        .build_call_direct(constructor_func_id, arg_regs, IrType::Void);
                }

                return Some(obj_ptr);
            }
        }

        // CLASS TYPE CONSTRUCTOR:
        // Allocate object on HEAP (not stack) since objects may escape the current function
        // When a method returns `new Foo()`, the object must outlive the callee's stack frame
        let _class_mir_type = self.convert_type(*class_type);

        // Look up pre-computed allocation size from register_class_metadata.
        // Key order: same-context SymbolId, then QUALIFIED NAME — the only
        // key stable across compilation contexts. Context-local TypeIds
        // (including accumulated cross-context TypeId→SymbolId maps) must
        // never be consulted: a shifted id resolves to another class's
        // size and silently under-allocates.
        let alloc_dbg = std::env::var_os("RAYZOR_ALLOC_DEBUG").is_some();
        let stage = std::cell::Cell::new("sym");
        let class_sym_for_name = actual_symbol_id.or_else(|| {
            self.type_table
                .get(*class_type)
                .and_then(|t| match &t.kind {
                    crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    _ => None,
                })
        });
        let class_qname: Option<&str> = class_sym_for_name
            .and_then(|sid| self.symbol_table.get_symbol(sid))
            .and_then(|sym| {
                sym.qualified_name
                    .and_then(|n| self.string_interner.get(n))
                    .or_else(|| self.string_interner.get(sym.name))
            });
        let obj_size: u64 = actual_symbol_id
            .and_then(|sid| self.class_alloc_sizes.get(&sid).copied())
            .or_else(|| {
                stage.set("name");
                class_qname.and_then(|n| self.class_alloc_sizes_by_name.get(n).copied())
            })
            // Fallback: declared instance-field count from the parsed
            // AST (class compiles later than this module — no layout
            // registered anywhere yet).
            .or_else(|| {
                stage.set("ast");
                let index = self.static_sig_index.as_ref()?.clone();
                let sym = self.symbol_table.get_symbol(class_sym_for_name?)?;
                let qname = sym.qualified_name.and_then(|n| self.string_interner.get(n));
                let bare = self.string_interner.get(sym.name);
                let mut index = index.borrow_mut();
                let no_parse = |_: &str| -> Option<std::path::PathBuf> { None };
                let count = qname
                    .and_then(|q| index.instance_field_count(q, &no_parse))
                    .or_else(|| {
                        bare.filter(|b| qname != Some(*b))
                            .and_then(|b| index.instance_field_count(b, &no_parse))
                    })?;
                Some((count as u64 + 1) * 8)
            })
            .unwrap_or_else(|| {
                stage.set("argcount");
                ((args.len() as u64 + 1) * 8).max(16)
            });
        if alloc_dbg {
            let cname = actual_symbol_id
                .and_then(|sid| self.symbol_table.get_symbol(sid))
                .and_then(|sym| {
                    sym.qualified_name
                        .and_then(|n| self.string_interner.get(n))
                        .or_else(|| self.string_interner.get(sym.name))
                })
                .unwrap_or("<?>");
            eprintln!(
                "[alloc-size] class={} sym={:?} type={:?} stage={} size={}",
                cname,
                actual_symbol_id,
                class_type,
                stage.get(),
                obj_size
            );
        }
        // Ensure the allocation covers inherited fields: a subclass of an
        // imported class (e.g. MyException extends haxe.Exception) can be
        // registered with an undersized `obj_size` because the parent's
        // fields weren't available at registration; inherited-field writes
        // (the `stack` field at `throw`) would then overflow the heap block.
        let obj_size = self.alloc_size_with_inheritance(*class_type, actual_symbol_id, obj_size);
        // Use heap allocation (malloc) for class instances
        let obj_ptr = self.build_heap_alloc(obj_size);
        let obj_ptr = obj_ptr?;

        // Store object header: runtime type_id at GEP index 0.
        // Use the same class-id resolver as typed throw/catch (without +1000).
        {
            // Store the runtime_type_id directly (no -1000 transform).
            // The id is now a stable name-hash; cast/is checks
            // compare with the same `runtime_type_id()` value, so
            // any offset transformation here would just have to be
            // mirrored on the comparison side. Keeping the id raw
            // makes both sides agree by construction.
            //
            // If `class_type` is a Placeholder (cross-module import),
            // `runtime_type_id` returns 0. Prefer the name-recovered
            // `actual_symbol_id`'s deterministic id so the header
            // carries the real class id (needed for is/cast/vtable and
            // for the interface fat-ptr wrap to find the vtable).
            let raw_type_id = self.runtime_type_id(*class_type);
            let runtime_type_id = if raw_type_id != 0 {
                raw_type_id
            } else {
                actual_symbol_id
                    .and_then(|sid| self.deterministic_class_type_id(sid))
                    .unwrap_or(raw_type_id)
            } as i64;
            if let Some(type_id_const) = self.builder.build_const(IrValue::I64(runtime_type_id)) {
                if let Some(index_0) = self.builder.build_const(IrValue::I32(0)) {
                    if let Some(header_ptr) =
                        self.builder.build_gep(obj_ptr, vec![index_0], IrType::I64)
                    {
                        self.builder.build_store(header_ptr, type_id_const);
                    }
                }
            }
        }

        // @:derive(Default): initialize all fields before constructor call.
        // Uses @:default(value) metadata if present, else type-based defaults.
        if let Some(class_sym) = actual_symbol_id {
            if self.derive_default_classes.contains(&class_sym) {
                if let Some(fields) = self.class_instance_fields.get(&class_sym).cloned() {
                    // Pre-collect field defaults to avoid borrow issues
                    let field_defaults: Vec<_> = fields
                        .iter()
                        .map(|&(field_sym, _, _)| self.field_default_exprs.get(&field_sym).cloned())
                        .collect();

                    for (i, &(_field_sym, field_type_id, gep_idx)) in fields.iter().enumerate() {
                        let field_ir_type = self.convert_type(field_type_id);
                        let default_val = if let Some(ref default_expr) = field_defaults[i] {
                            self.lower_expression(default_expr)
                                .or_else(|| self.build_type_default(field_type_id))
                        } else {
                            self.build_type_default(field_type_id)
                        };
                        if let Some(val) = default_val {
                            let idx = self.builder.build_const(IrValue::I32(gep_idx as i32));
                            if let Some(idx) = idx {
                                let ptr = self.builder.build_gep(obj_ptr, vec![idx], field_ir_type);
                                if let Some(ptr) = ptr {
                                    self.builder.build_store(ptr, val);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Look up constructor by TypeId - use the resolved constructor_type_id
        let constructor_func_id = self.constructor_map.get(&constructor_type_id).copied();
        if std::env::var("RAYZOR_CTOR_DEBUG").is_ok() {
            let cname = debug_class_name.unwrap_or("?");
            eprintln!(
                "[CTOR] class={} path={} class_type={:?} resolved_type={:?} fid={:?}",
                cname, ctor_path, class_type, constructor_type_id, constructor_func_id
            );
        }
        if let Some(constructor_func_id) = constructor_func_id {
            // Call constructor with object as first argument.
            //
            // Each user arg goes through `maybe_materialize_for_call`
            // (handles anon-views, class→anon coercion, and the
            // class→interface fat-pointer wrap). Without this,
            // `new Holder(c)` where `Holder.new(it:I)` stores the
            // raw class pointer into the interface-typed field and
            // `h.slot.value()` SIGSEGVs on virtual dispatch.
            let mut arg_regs: Vec<IrId> = Vec::with_capacity(args.len() + 1);
            arg_regs.push(obj_ptr);
            // HIR constructor params don't include `this`, so user arg
            // `args[i]` lines up with HIR param index `i`. Don't shift
            // by +1 the way Call-method dispatch does — that would
            // bias the lookup off the end of `function_param_hir_types`
            // and silently skip Path 3 (class→interface wrap).
            for (i, a) in args.iter().enumerate() {
                if let Some(reg) = self.lower_expression(a) {
                    let wrapped =
                        self.maybe_materialize_for_call(a, reg, Some(constructor_func_id), i);
                    arg_regs.push(wrapped);
                }
            }

            let pre_fill = arg_regs.len();
            // Coerce Int→Float at cross-module call boundaries
            self.coerce_args_for_cross_module_call(constructor_func_id, &mut arg_regs, true);
            // Fill in default values for any missing optional parameters
            self.fill_default_args(constructor_func_id, &mut arg_regs, true);
            let post_fill = arg_regs.len();
            let sig_params = self
                .builder
                .module
                .functions
                .get(&constructor_func_id)
                .map(|f| f.signature.parameters.len())
                .unwrap_or(0);

            // Generic constructor specialization: a `new Foo<T1,T2>(...)`
            // call must carry the concrete type args so the monomorphizer
            // can specialize Foo.new. Untyped constructor params lower to
            // `*void`, which defeats the monomorphizer's TypeVar-based
            // argument inference (extract_generic_call), so without explicit
            // type_args the generic ctor template is left un-monomorphized
            // and trap-stubbed → SIGILL at runtime. This bit haxe.ds.TreeNode
            // (self-referential `left`/`right` + an in-ctor method call).
            // Prefer the explicit `new Foo<...>` args; fall back to the
            // class_type's own type_args (GenericInstance / Class).
            let ctor_type_args: Vec<IrType> = {
                let mut ta_ids: Vec<TypeId> = hir_type_args.to_vec();
                if ta_ids.is_empty() {
                    if let Some(ti) = self.type_table.get(*class_type) {
                        ta_ids = match &ti.kind {
                            crate::tast::TypeKind::GenericInstance { type_args, .. }
                            | crate::tast::TypeKind::Class { type_args, .. } => type_args.clone(),
                            _ => Vec::new(),
                        };
                    }
                }
                let converted: Vec<IrType> = ta_ids.iter().map(|&t| self.convert_type(t)).collect();
                // Only request specialization when every arg is concrete; an
                // unresolved TypeVar means we're still inside a generic
                // context and the enclosing instantiation handles it.
                if converted.is_empty() || converted.iter().any(|t| matches!(t, IrType::TypeVar(_)))
                {
                    Vec::new()
                } else {
                    converted
                }
            };

            // Constructor returns void, so we ignore the result
            if ctor_type_args.is_empty() {
                self.builder
                    .build_call_direct(constructor_func_id, arg_regs, IrType::Void);
            } else {
                self.builder.build_call_direct_with_type_args(
                    constructor_func_id,
                    arg_regs,
                    IrType::Void,
                    ctor_type_args,
                );
            }

            // Transfer ownership: constructor args that are heap-allocated
            // are now owned by the new object (stored in its fields).
            // Remove them from drop tracking to prevent premature free.
            for arg in args.iter() {
                if let HirExprKind::Variable { symbol, .. } = &arg.kind {
                    if self.owned_heap_values.contains_key(symbol) {
                        self.owned_heap_values.remove(symbol);
                    }
                }
            }
        } else if let Some(ctor_fn) =
            self.resolve_cross_module_constructor_by_name(actual_symbol_id, final_class_name)
        {
            // Cross-module: the class's constructor isn't in this
            // context's `constructor_map`/`constructor_name_map` but is
            // resolvable by FQN (`<class>.new`) in the shared name map.
            // Call it so the object is constructed rather than left with
            // default-null fields.
            let mut arg_regs: Vec<IrId> = Vec::with_capacity(args.len() + 1);
            arg_regs.push(obj_ptr);
            for a in args.iter() {
                if let Some(reg) = self.lower_expression(a) {
                    arg_regs.push(reg);
                }
            }
            self.builder
                .build_call_direct(ctor_fn, arg_regs, IrType::Void);
            for arg in args.iter() {
                if let HirExprKind::Variable { symbol, .. } = &arg.kind {
                    self.owned_heap_values.remove(symbol);
                }
            }
        } else if let Some(ctor_key) =
            self.cross_module_constructor_fqn_key(actual_symbol_id, final_class_name)
        {
            // Cross-module, class compiles LATER in this import pass
            // (cycle-breaker / retry ordering), so its constructor isn't
            // resolvable by name yet either. Emit the call against a
            // named forward-ref stub; `fixup_stale_cross_module_refs`
            // redirects it to the real constructor by qualified name
            // once every module is loaded. Silently emitting NO call
            // here left the object zero-initialized.
            let mut arg_regs: Vec<IrId> = Vec::with_capacity(args.len() + 1);
            arg_regs.push(obj_ptr);
            for a in args.iter() {
                if let Some(reg) = self.lower_expression(a) {
                    arg_regs.push(reg);
                }
            }
            let param_types: Vec<IrType> = arg_regs
                .iter()
                .map(|r| {
                    self.builder
                        .get_register_type(*r)
                        .unwrap_or(IrType::Ptr(Box::new(IrType::Void)))
                })
                .collect();
            let stub_id =
                self.register_stdlib_mir_forward_ref(&ctor_key, param_types, IrType::Void);
            self.builder
                .build_call_direct(stub_id, arg_regs, IrType::Void);
            for arg in args.iter() {
                if let HirExprKind::Variable { symbol, .. } = &arg.kind {
                    self.owned_heap_values.remove(symbol);
                }
            }
        }

        if let Some(class_name) = final_class_name {
            self.register_class_hints
                .insert(obj_ptr, class_name.to_string());
        }
        Some(obj_ptr)
    }
}
