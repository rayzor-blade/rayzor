//! The generated module `__init__`: statics, reflect wrappers, runtime externs.

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
    pub(crate) fn generate_module_init_function(&mut self) {
        if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
            let mut names: Vec<(u32, String)> = self
                .builder
                .module
                .globals
                .values()
                .map(|g| (g.id.0, g.name.clone()))
                .collect();
            names.sort();
            eprintln!("[globals] module has {} global(s):", names.len());
            for (id, name) in &names {
                eprintln!("[globals]   @g{} {}", id, name);
            }
            let dyn_syms: Vec<String> = self
                .dynamic_globals
                .iter()
                .map(|(sym, _)| {
                    self.symbol_table
                        .get_symbol(*sym)
                        .and_then(|sy| self.string_interner.get(sy.name))
                        .unwrap_or("<?>")
                        .to_string()
                })
                .collect();
            eprintln!("[globals] __init__ will initialise: {:?}", dyn_syms);
        }
        // `__init__` materialises globals into backend storage: reset every
        // global to its declared initializer/default, then evaluate dynamic
        // initializers in source order. It runs at every module load so statics
        // behave like standard Haxe and never retain stale values.
        let init_sig = FunctionSignatureBuilder::new()
            .returns(IrType::Void)
            .calling_convention(CallingConvention::Haxe)
            .build();

        let init_symbol = SymbolId::from_raw(u32::MAX - 1); // Reserved symbol for __init__
        let _init_func_id =
            self.builder
                .start_function(init_symbol, "__init__".to_string(), init_sig);

        let saved_symbol_map = self.symbol_map.clone();
        // Per-function isolation: __init__ has its own SSA namespace; snapshot
        // and clear strict_move_locals in parallel with symbol_map.
        let saved_strict_move_locals = self.strict_move_locals.clone();
        self.symbol_map.clear();
        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        self.strict_move_locals.clear();

        let globals_to_reset: Vec<(IrGlobalId, IrType, IrValue)> = self
            .builder
            .module
            .globals
            .values()
            .map(|global| {
                (
                    global.id,
                    global.ty.clone(),
                    self.reset_value_for_global(global),
                )
            })
            .collect();

        for (global_id, global_ty, reset_value) in globals_to_reset {
            if let Some(val_reg) = self.builder.build_const(reset_value) {
                let store_val = match self.builder.get_register_type(val_reg) {
                    Some(val_ty) if val_ty != global_ty => self
                        .builder
                        .build_cast(val_reg, val_ty, global_ty.clone())
                        .unwrap_or(val_reg),
                    _ => val_reg,
                };
                self.builder.build_store_global(global_id, store_val);
            }
        }

        // Initialize dynamic globals (non-constant initializers).
        for (symbol, init_expr) in &self.dynamic_globals.clone() {
            // Try constant folding first for simple expressions
            let const_val = self.try_evaluate_constant_init(init_expr);

            let global_id = self.global_symbol_map.get(symbol).copied().or_else(|| {
                let sym_name = self
                    .symbol_table
                    .get_symbol(*symbol)
                    .and_then(|s| self.string_interner.get(s.name));
                sym_name.and_then(|name| {
                    self.builder
                        .module
                        .globals
                        .values()
                        .find(|g| g.name.ends_with(&format!(".{}", name)) || g.name == name)
                        .map(|g| g.id)
                })
            });

            if let Some(gid) = global_id {
                if let Some(cv) = const_val {
                    if let Some(val_reg) = self.builder.build_const(cv) {
                        let global_ty = self.builder.module.globals.get(&gid).map(|g| g.ty.clone());
                        let store_val = if let Some(ref gty) = global_ty {
                            let val_ty = self.builder.get_register_type(val_reg);
                            if val_ty.as_ref() != Some(gty) && val_ty.is_some() {
                                self.builder
                                    .build_cast(val_reg, val_ty.unwrap(), gty.clone())
                                    .unwrap_or(val_reg)
                            } else {
                                val_reg
                            }
                        } else {
                            val_reg
                        };
                        self.builder.build_store_global(gid, store_val);
                    }
                } else if let Some(init_value) = self.lower_expression(init_expr) {
                    self.builder.build_store_global(gid, init_value);
                }
            }
        }

        self.builder.build_return(None);
        self.builder.finish_function();

        self.symbol_map = saved_symbol_map;
        self.strict_move_locals = saved_strict_move_locals;
    }

    pub(crate) fn generate_constructor_reflect_wrappers(&mut self) {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let ptr_void = IrType::Ptr(Box::new(IrType::Void));

        // Runtime helper used by generated wrappers.
        let array_get_i64_fn = self.get_or_register_extern_function(
            "haxe_array_get_i64",
            vec![ptr_void.clone(), IrType::I64],
            IrType::I64,
        );

        let ctor_entries: Vec<(TypeId, IrFunctionId)> = self
            .constructor_map
            .iter()
            .map(|(class_type_id, ctor_func_id)| (*class_type_id, *ctor_func_id))
            .collect();

        let saved_symbol_map = self.symbol_map.clone();
        // Per-function isolation: each reflect wrapper has its own SSA namespace;
        // snapshot and clear strict_move_locals in parallel with symbol_map.
        let saved_strict_move_locals = self.strict_move_locals.clone();
        self.symbol_map.clear();
        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        self.strict_move_locals.clear();

        for (class_type_id, ctor_func_id) in ctor_entries {
            if self
                .constructor_reflect_wrappers
                .contains_key(&class_type_id)
            {
                continue;
            }

            // Only emit wrappers for classes known to this module's type metadata.
            let has_typedef = self
                .builder
                .module
                .types
                .values()
                .any(|typedef| typedef.type_id == class_type_id);
            if !has_typedef {
                continue;
            }

            let ctor_sig = match self.builder.module.functions.get(&ctor_func_id) {
                Some(func) => func.signature.clone(),
                None => continue, // Imported constructor body not available in this module.
            };
            if ctor_sig.parameters.is_empty() {
                continue;
            }

            let wrapper_symbol = SymbolId::from_raw(u32::MAX - 1000 - self.next_wrapper_id);
            self.next_wrapper_id += 1;
            let wrapper_name = format!("__reflect_ctor_wrap_{}", class_type_id.as_raw());

            let wrapper_sig = FunctionSignatureBuilder::new()
                .param("obj".to_string(), ptr_u8.clone())
                .param("args".to_string(), ptr_void.clone())
                .returns(IrType::Void)
                .calling_convention(CallingConvention::Haxe)
                .build();

            let wrapper_func_id =
                self.builder
                    .start_function(wrapper_symbol, wrapper_name, wrapper_sig);

            let (obj_reg, args_reg) = {
                let Some(func) = self.builder.current_function() else {
                    self.builder.finish_function();
                    continue;
                };
                let Some(obj) = func.get_param_reg(0) else {
                    self.builder.finish_function();
                    continue;
                };
                let Some(args) = func.get_param_reg(1) else {
                    self.builder.finish_function();
                    continue;
                };
                (obj, args)
            };

            let mut ctor_args: Vec<IrId> = Vec::new();
            let this_ty = ctor_sig.parameters[0].ty.clone();
            let this_arg = if this_ty == ptr_u8 {
                obj_reg
            } else {
                self.builder
                    .build_cast(obj_reg, ptr_u8.clone(), this_ty)
                    .unwrap_or(obj_reg)
            };
            ctor_args.push(this_arg);

            for (param_idx, param) in ctor_sig.parameters.iter().enumerate().skip(1) {
                let Some(index_reg) = self
                    .builder
                    .build_const(IrValue::I64((param_idx.saturating_sub(1)) as i64))
                else {
                    continue;
                };

                let raw_i64 = self
                    .builder
                    .build_call_direct(array_get_i64_fn, vec![args_reg, index_reg], IrType::I64)
                    .or_else(|| self.builder.build_const(IrValue::I64(0)));
                let Some(raw_i64) = raw_i64 else {
                    continue;
                };

                let arg_reg_opt = match &param.ty {
                    IrType::Bool => self.builder.build_cast(raw_i64, IrType::I64, IrType::Bool),
                    IrType::F64 => self.builder.build_bitcast(raw_i64, IrType::F64),
                    IrType::F32 => {
                        let as_i32 = self.builder.build_cast(raw_i64, IrType::I64, IrType::I32);
                        as_i32.and_then(|v| self.builder.build_bitcast(v, IrType::F32))
                    }
                    IrType::I8
                    | IrType::I16
                    | IrType::I32
                    | IrType::I64
                    | IrType::U8
                    | IrType::U16
                    | IrType::U32
                    | IrType::U64 => {
                        if param.ty == IrType::I64 {
                            Some(raw_i64)
                        } else {
                            self.builder
                                .build_cast(raw_i64, IrType::I64, param.ty.clone())
                        }
                    }
                    _ => {
                        let ref_ptr = self
                            .builder
                            .build_cast(raw_i64, IrType::I64, ptr_u8.clone());
                        match &param.ty {
                            IrType::Ptr(_) | IrType::Ref(_) | IrType::Any => {
                                ref_ptr.and_then(|r| {
                                    self.builder.build_cast(r, ptr_u8.clone(), param.ty.clone())
                                })
                            }
                            IrType::String | IrType::Function { .. } => ref_ptr,
                            _ => ref_ptr,
                        }
                    }
                };

                if let Some(arg_reg) = arg_reg_opt {
                    ctor_args.push(arg_reg);
                }
            }

            self.builder
                .build_call_direct(ctor_func_id, ctor_args, IrType::Void);
            self.builder.build_return(None);
            self.builder.finish_function();

            self.constructor_reflect_wrappers
                .insert(class_type_id, wrapper_func_id);
        }

        self.symbol_map = saved_symbol_map;
        self.strict_move_locals = saved_strict_move_locals;
    }

    /// Ensure Future runtime extern functions are declared in the current module.
    /// Returns (create_id, await_id, poll_id, is_ready_id).
    pub(crate) fn ensure_future_externs(
        &mut self,
    ) -> (IrFunctionId, IrFunctionId, IrFunctionId, IrFunctionId) {
        use crate::ir::modules::IrExternFunction;

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        let externs_to_declare: Vec<(&str, Vec<IrType>, IrType)> = vec![
            (
                "rayzor_future_create",
                vec![ptr_u8.clone(), ptr_u8.clone()],
                ptr_u8.clone(),
            ),
            ("rayzor_future_await", vec![ptr_u8.clone()], ptr_u8.clone()),
            ("rayzor_future_poll", vec![ptr_u8.clone()], ptr_u8.clone()),
            ("rayzor_future_is_ready", vec![ptr_u8.clone()], IrType::Bool),
        ];

        let mut ids = Vec::new();
        for (name, params, ret) in externs_to_declare {
            let existing = self
                .builder
                .module
                .extern_functions
                .iter()
                .find(|(_, f)| f.name == name)
                .map(|(id, _)| *id);
            if let Some(id) = existing {
                ids.push(id);
                continue;
            }
            let id = self.builder.module.alloc_function_id();
            let sig = crate::ir::IrFunctionSignature {
                parameters: params
                    .into_iter()
                    .enumerate()
                    .map(|(i, ty)| crate::ir::functions::IrParameter {
                        name: format!("p{}", i),
                        ty,
                        reg: IrId(i as u32),
                        by_ref: false,
                    })
                    .collect(),
                return_type: ret,
                calling_convention: crate::ir::CallingConvention::C,
                can_throw: false,
                type_params: Vec::new(),
                uses_sret: false,
            };
            self.builder.module.extern_functions.insert(
                id,
                IrExternFunction {
                    id,
                    name: name.to_string(),
                    symbol_id: SymbolId::from_raw(9999),
                    signature: sig,
                    source: "runtime".to_string(),
                },
            );
            ids.push(id);
        }

        (ids[0], ids[1], ids[2], ids[3])
    }

    pub(crate) fn ensure_terminator(&mut self) {
        let is_term = self.is_terminated();
        if !is_term {
            self.builder.build_return(None);
        }
    }

    /// Register a heap-allocated value as owned by a variable
    /// This is called when a variable is assigned a newly allocated value (from `new`)
    pub(crate) fn register_owned_value(&mut self, symbol: SymbolId, ir_id: IrId) {
        if let Some(old_ir_id) = self.owned_heap_values.get(&symbol).copied() {
            // Reassignment frees the old value, but only when its definition
            // dominates this block: `var x; if c { x = A } else { x = B }`
            // lowers the arms sequentially, so the else arm still sees A in the
            // tracker even though A never runs on this path.
            self.emit_tracked_free(old_ir_id, true);

            // Scope entries keep pointing at the ORIGINAL declaration, which may
            // live in a different block. Reassigned values are freed here rather
            // than at scope exit, so mark the symbol for scope exit to skip.
            self.reassigned_in_scope.insert(symbol);
        } else {
            // New declaration - add to current scope for cleanup on scope exit
            if let Some(current_scope) = self.drop_scope_stack.last_mut() {
                current_scope.push((symbol, ir_id));
            }
        }

        self.owned_heap_values.insert(symbol, ir_id);

        // Track Drop class association for @:derive(Drop)
        if !self.derive_drop_classes.is_empty() {
            if let Some(class_name) = self.register_class_hints.get(&ir_id).cloned() {
                for &class_sym in &self.derive_drop_classes {
                    if let Some(symbol_info) = self.symbol_table.get_symbol(class_sym) {
                        if self.string_interner.get(symbol_info.name) == Some(class_name.as_str()) {
                            self.ir_to_drop_class.insert(ir_id, class_sym);
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Register a temporary heap-allocated value that needs dropping after use
    /// This is for intermediate results like `new Complex(...).mul(...)`
    pub(crate) fn register_temp_value(&mut self, ir_id: IrId) {
        self.temp_heap_values.push(ir_id);
    }

    /// Lazily initialize TCC runtime extern function declarations
    pub(crate) fn ensure_tcc_runtime(&mut self) -> TccFuncIds {
        if let Some(ids) = self.tcc_func_ids {
            return ids;
        }

        let create = self.declare_tcc_extern("rayzor_tcc_create", vec![], IrType::I64, 0);
        let compile = self.declare_tcc_extern(
            "rayzor_tcc_compile",
            vec![IrType::I64, IrType::I64],
            IrType::I32,
            1,
        );
        let add_value_symbol = self.declare_tcc_extern(
            "rayzor_tcc_add_value_symbol",
            vec![IrType::I64, IrType::I64, IrType::I64],
            IrType::I64,
            2,
        );
        let relocate =
            self.declare_tcc_extern("rayzor_tcc_relocate", vec![IrType::I64], IrType::I32, 3);
        let get_symbol = self.declare_tcc_extern(
            "rayzor_tcc_get_symbol",
            vec![IrType::I64, IrType::I64],
            IrType::I64,
            4,
        );
        let delete =
            self.declare_tcc_extern("rayzor_tcc_delete", vec![IrType::I64], IrType::Void, 5);
        let call0 = self.declare_tcc_extern("rayzor_tcc_call0", vec![IrType::I64], IrType::I64, 6);
        let free_value =
            self.declare_tcc_extern("rayzor_tcc_free_value", vec![IrType::I64], IrType::Void, 7);
        let string_ptr = IrType::Ptr(Box::new(IrType::String));
        let string_replace = self.declare_tcc_extern(
            "haxe_string_replace",
            vec![string_ptr.clone(), string_ptr.clone(), string_ptr.clone()],
            string_ptr,
            8,
        );

        let add_framework = self.declare_tcc_extern(
            "rayzor_tcc_add_framework",
            vec![IrType::I64, IrType::I64],
            IrType::I32,
            9,
        );
        let add_include_path = self.declare_tcc_extern(
            "rayzor_tcc_add_include_path",
            vec![IrType::I64, IrType::I64],
            IrType::I32,
            10,
        );
        let add_file = self.declare_tcc_extern(
            "rayzor_tcc_add_file",
            vec![IrType::I64, IrType::I64],
            IrType::I32,
            11,
        );
        let add_clib = self.declare_tcc_extern(
            "rayzor_tcc_add_clib",
            vec![IrType::I64, IrType::I64],
            IrType::I32,
            12,
        );

        let ids = TccFuncIds {
            create,
            compile,
            add_value_symbol,
            relocate,
            get_symbol,
            delete,
            call0,
            free_value,
            string_replace,
            add_framework,
            add_include_path,
            add_file,
            add_clib,
        };
        self.tcc_func_ids = Some(ids);
        ids
    }

    pub(crate) fn register_enum_metadata(&mut self, _type_id: TypeId, enum_decl: &HirEnum) {
        // Register enum type with discriminant values
        let typedef_id = self.builder.module.alloc_typedef_id();

        // Use the symbol's type_id for consistency with trace call sites
        let sym_type_id = self
            .symbol_table
            .get_symbol(enum_decl.symbol_id)
            .map(|sym| sym.type_id)
            .unwrap_or(_type_id);

        let enum_name = self
            .string_interner
            .get(enum_decl.name)
            .unwrap_or("<unknown>")
            .to_string();

        let mut variants = Vec::new();
        for (i, variant) in enum_decl.variants.iter().enumerate() {
            let discriminant = variant.discriminant.unwrap_or(i as i32) as i64;

            let variant_name = self
                .string_interner
                .get(variant.name)
                .unwrap_or("<unknown>")
                .to_string();

            let fields: Vec<IrField> = variant
                .fields
                .iter()
                .map(|field| {
                    let field_name = self
                        .string_interner
                        .get(field.name)
                        .unwrap_or("<unknown>")
                        .to_string();
                    IrField {
                        name: field_name,
                        ty: self.convert_type(field.ty),
                        offset: None,
                    }
                })
                .collect();

            variants.push(IrEnumVariant {
                name: variant_name,
                discriminant,
                fields,
            });
        }

        let enum_runtime_id = self.deterministic_iface_or_enum_type_id(enum_decl.symbol_id, "enum");
        let typedef = IrTypeDef {
            id: typedef_id,
            name: enum_name,
            type_id: sym_type_id,
            runtime_type_id: enum_runtime_id,
            definition: IrTypeDefinition::Enum {
                variants,
                discriminant_type: IrType::I32,
            },
            source_location: IrSourceLocation::unknown(),
            super_type_id: None,
        };

        self.builder.module.add_type(typedef);
    }

    pub(crate) fn register_type_metadata(&mut self, type_id: TypeId, type_decl: &HirTypeDecl) {
        self.dbg_type_meta_calls += 1;
        // MIR type definitions carry the runtime type information: enum
        // discriminants for pattern matching, struct field layouts for field
        // access, interface method tables for dynamic dispatch, and the data
        // behind runtime type checks.
        match type_decl {
            HirTypeDecl::Class(class) => {
                self.register_class_metadata(type_id, class);
            }
            HirTypeDecl::Interface(interface) => {
                self.register_interface_metadata(type_id, interface);
            }
            HirTypeDecl::Enum(enum_decl) => {
                self.register_enum_metadata(type_id, enum_decl);
            }
            HirTypeDecl::Abstract(abstract_decl) => {
                self.register_abstract_metadata(type_id, abstract_decl);
            }
            HirTypeDecl::TypeAlias(alias) => {
                self.register_alias_metadata(type_id, alias);
            }
        }
    }
}
