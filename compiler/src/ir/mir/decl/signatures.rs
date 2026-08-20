//! Signature registration — pass 1, before bodies exist.

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
    /// Register a function signature without lowering the body (Pass 1)
    /// This creates the function stub and adds it to function_map
    pub(crate) fn register_function_signature(
        &mut self,
        symbol_id: SymbolId,
        hir_func: &HirFunction,
        this_type: Option<TypeId>,
    ) {
        self.record_param_ownership(symbol_id, hir_func);
        self.record_consume_method(symbol_id, hir_func);
        let mut signature = self.build_function_signature(hir_func);

        // 'this' is always a pointer to the instance, generic parameters or not.
        if let Some(type_id) = this_type {
            let this_type = match self.convert_type(type_id) {
                IrType::Ptr(_) => IrType::Ptr(Box::new(IrType::Void)),
                // Unresolved (a generic class with no instantiation) is a pointer too.
                _ => IrType::Ptr(Box::new(IrType::Void)),
            };
            signature.parameters.insert(
                0,
                IrParameter {
                    name: "this".to_string(),
                    ty: this_type,
                    reg: IrId::new(0), // Will be properly assigned when body is lowered
                    by_ref: false,
                },
            );
        }

        let func_name = self
            .string_interner
            .get(hir_func.name)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("func_{}", symbol_id.as_raw()));

        let func_id = self.builder.start_function(symbol_id, func_name, signature);
        self.function_map.insert(symbol_id, func_id);

        // A bodyless declaration (an `extern class Tensor` method like `addInto`)
        // becomes an empty forward-ref placeholder, and `IrFunction::new` defaults
        // `kind` to `UserDefined`. `own_func_ids` (compilation.rs) reads
        // `kind != MirWrapper` as "this compile owns the function" and protects it
        // from the stdlib merge's by-name replacement, so a mistagged stdlib wrapper
        // survives the merge empty and traps when called. The tag must be set here at
        // creation: a file that declares the extern but never calls it never reaches
        // the call-site fixup in `register_stdlib_mir_forward_ref`.
        if hir_func.body.is_none() {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                if self.stdlib_mapping.is_mir_wrapper_function(&func.name) {
                    func.kind = crate::ir::FunctionKind::MirWrapper;
                }
            }
        }

        if hir_func.is_keep {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                func.attributes
                    .custom
                    .insert("keep".to_string(), "true".to_string());
            }
        }

        // Store default parameter expressions for call-site filling
        let defaults: Vec<Option<HirExpr>> =
            hir_func.params.iter().map(|p| p.default.clone()).collect();
        if defaults.iter().any(|d| d.is_some()) {
            self.function_param_defaults.insert(func_id, defaults);
        }

        // Store parameter HIR types for structural subtyping materialization at call sites
        let param_types: Vec<TypeId> = hir_func.params.iter().map(|p| p.ty).collect();
        self.function_param_hir_types.insert(func_id, param_types);

        // Record constrained type parameter info for call-site fat pointer wrapping
        self.record_constrained_params(func_id, hir_func, this_type.is_some());

        // Respect Haxe `inline` keyword — always inline these functions
        if hir_func.is_inline {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                func.attributes.inline = crate::ir::InlineHint::Always;
            }
        }

        self.propagate_js_import(symbol_id, func_id, this_type);

        self.check_move_flow();
        self.builder.finish_function();
    }

    /// Register a function signature with class type parameters (for generic class methods)
    /// This version includes the class's type parameters in the function signature
    pub(crate) fn register_function_signature_with_class_type_params(
        &mut self,
        symbol_id: SymbolId,
        hir_func: &HirFunction,
        this_type: Option<TypeId>,
        class_type_params: &[HirTypeParam],
    ) {
        self.record_param_ownership(symbol_id, hir_func);
        self.record_consume_method(symbol_id, hir_func);
        let mut signature =
            self.build_function_signature_with_class_type_params(hir_func, class_type_params);

        if let Some(type_id) = this_type {
            // Distinguish abstracts from classes off the HIR TypeKind: convert_type()
            // can misresolve class types to I32 in cross-module compilation when the
            // type table entry is missing.
            let is_abstract_value_type = {
                let type_table = self.type_table;
                match type_table.get(type_id).map(|t| &t.kind) {
                    Some(crate::tast::TypeKind::Abstract {
                        underlying: Some(u),
                        ..
                    }) => {
                        let u_ty = self.convert_type(*u);
                        matches!(
                            u_ty,
                            IrType::I8
                                | IrType::I16
                                | IrType::I32
                                | IrType::I64
                                | IrType::U8
                                | IrType::U16
                                | IrType::U32
                                | IrType::U64
                                | IrType::F32
                                | IrType::F64
                                | IrType::Bool
                        )
                    }
                    _ => false, // Classes, interfaces, etc. always use pointer
                }
            };
            // TypeIds are context-local and drift (adding one type to StdTypes shifts
            // every TypeId while SymbolIds stay put), so a class's id can land on an
            // abstract-over-Int and the check above would then pass `this` BY VALUE,
            // truncating the 64-bit receiver. The HIR module being lowered is
            // authoritative about what it declares, so let it veto: if HIR says this
            // id is a class, it is a class, whatever the type_table entry says.
            let hir_says_class = matches!(
                self.current_hir_types.get(&type_id),
                Some(HirTypeDecl::Class(_))
            );
            let is_abstract_value_type = is_abstract_value_type && !hir_says_class;
            let this_ir_type = if is_abstract_value_type {
                self.convert_type(type_id) // Raw value type for abstract underlying types
            } else {
                IrType::Ptr(Box::new(IrType::Void)) // Heap pointer for classes
            };
            signature.parameters.insert(
                0,
                IrParameter {
                    name: "this".to_string(),
                    ty: this_ir_type,
                    reg: IrId::new(0),
                    by_ref: false,
                },
            );
        }

        let func_name = self
            .string_interner
            .get(hir_func.name)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("func_{}", symbol_id.as_raw()));

        let func_id = self.builder.start_function(symbol_id, func_name, signature);
        self.function_map.insert(symbol_id, func_id);

        // See the identical guard in `register_function_signature` for why
        // a bodyless stdlib-mapped extern method must be retagged
        // `MirWrapper` here at creation, not left to a later call-site fixup.
        if hir_func.body.is_none() {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                if self.stdlib_mapping.is_mir_wrapper_function(&func.name) {
                    func.kind = crate::ir::FunctionKind::MirWrapper;
                }
            }
        }

        if hir_func.is_keep {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                func.attributes
                    .custom
                    .insert("keep".to_string(), "true".to_string());
            }
        }

        // Store default parameter expressions for call-site filling
        let defaults: Vec<Option<HirExpr>> =
            hir_func.params.iter().map(|p| p.default.clone()).collect();
        if defaults.iter().any(|d| d.is_some()) {
            self.function_param_defaults.insert(func_id, defaults);
        }

        // Store parameter HIR types for structural subtyping materialization at call sites
        let param_types: Vec<TypeId> = hir_func.params.iter().map(|p| p.ty).collect();
        self.function_param_hir_types.insert(func_id, param_types);

        // Record constrained type parameter info for call-site fat pointer wrapping
        self.record_constrained_params(func_id, hir_func, this_type.is_some());

        // Respect Haxe `inline` keyword — always inline these functions
        if hir_func.is_inline {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                func.attributes.inline = crate::ir::InlineHint::Always;
            }
        }

        self.propagate_js_import(symbol_id, func_id, this_type);

        self.check_move_flow();
        self.builder.finish_function();
    }

    /// Register constructor signature (Pass 1)
    pub(crate) fn register_constructor_signature(
        &mut self,
        class_symbol: SymbolId,
        constructor: &HirConstructor,
        type_id: TypeId,
    ) {
        let class_name = self
            .symbol_table
            .get_symbol(class_symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("?");
        // 'this' is always a pointer to the instance, generic parameters or not.
        let this_type = match self.convert_type(type_id) {
            IrType::Ptr(_) => IrType::Ptr(Box::new(IrType::Void)),
            _ => IrType::Ptr(Box::new(IrType::Void)),
        };
        let mut sig_builder = FunctionSignatureBuilder::new().param("this".to_string(), this_type);

        for param in &constructor.params {
            let param_name = self
                .string_interner
                .get(param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("param_{}", param.symbol_id.as_raw()));
            let ir_type = self.convert_type(param.ty);
            sig_builder = sig_builder.param(param_name, ir_type);
        }

        let mut signature = sig_builder.returns(IrType::Void).build();

        for (i, param) in signature.parameters.iter_mut().enumerate() {
            param.reg = IrId::new(i as u32);
        }

        let func_id = self
            .builder
            .start_function(class_symbol, "new".to_string(), signature);
        self.function_map.insert(class_symbol, func_id);
        self.constructor_map.insert(type_id, func_id);
        self.register_constructor_by_name(class_symbol, func_id);

        // Store default parameter expressions for call-site filling
        let defaults: Vec<Option<HirExpr>> = constructor
            .params
            .iter()
            .map(|p| p.default.clone())
            .collect();
        if defaults.iter().any(|d| d.is_some()) {
            self.function_param_defaults.insert(func_id, defaults);
        }

        // Store parameter HIR types for structural subtyping materialization at call sites
        let param_types: Vec<TypeId> = constructor.params.iter().map(|p| p.ty).collect();
        self.function_param_hir_types.insert(func_id, param_types);

        // Fallback key: a TypeId derived from the class SymbolId.
        let fallback_type_id = TypeId::from_raw(class_symbol.as_raw());
        if fallback_type_id != type_id {
            self.constructor_map.insert(fallback_type_id, func_id);
        }

        self.check_move_flow();
        self.builder.finish_function();
    }

    /// Register constructor signature with class type params (for generic classes)
    pub(crate) fn register_constructor_signature_with_class_type_params(
        &mut self,
        class_symbol: SymbolId,
        constructor: &HirConstructor,
        type_id: TypeId,
        class_type_params: &[HirTypeParam],
    ) {
        let this_type = match self.convert_type(type_id) {
            IrType::Ptr(_) => IrType::Ptr(Box::new(IrType::Void)),
            _ => IrType::Ptr(Box::new(IrType::Void)),
        };
        let mut sig_builder = FunctionSignatureBuilder::new().param("this".to_string(), this_type);

        for type_param in class_type_params {
            let param_name = self
                .string_interner
                .get(type_param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("T{}", type_param.name.as_raw()));
            sig_builder = sig_builder.type_param(param_name);
        }

        for param in &constructor.params {
            let param_name = self
                .string_interner
                .get(param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("param_{}", param.symbol_id.as_raw()));
            let ir_ty = self.convert_type(param.ty);
            let kind = self
                .type_table
                .get(param.ty)
                .map(|t| format!("{:?}", t.kind));
            sig_builder = sig_builder.param(param_name, ir_ty);
        }

        let mut signature = sig_builder.returns(IrType::Void).build();

        for (i, param) in signature.parameters.iter_mut().enumerate() {
            param.reg = IrId::new(i as u32);
        }

        let func_id = self
            .builder
            .start_function(class_symbol, "new".to_string(), signature);
        self.function_map.insert(class_symbol, func_id);
        self.constructor_map.insert(type_id, func_id);
        self.register_constructor_by_name(class_symbol, func_id);

        // Store default parameter expressions for call-site filling
        let defaults: Vec<Option<HirExpr>> = constructor
            .params
            .iter()
            .map(|p| p.default.clone())
            .collect();
        if defaults.iter().any(|d| d.is_some()) {
            self.function_param_defaults.insert(func_id, defaults);
        }

        // Store parameter HIR types for structural subtyping materialization at call sites
        let param_types: Vec<TypeId> = constructor.params.iter().map(|p| p.ty).collect();
        self.function_param_hir_types.insert(func_id, param_types);

        let fallback_type_id = TypeId::from_raw(class_symbol.as_raw());
        if fallback_type_id != type_id {
            self.constructor_map.insert(fallback_type_id, func_id);
        }

        self.check_move_flow();
        self.builder.finish_function();
    }

    /// Register a constructor by qualified name for cross-file resolution.
    /// The same class gets a different TypeId in each file that loads it, so the
    /// qualified name is the only stable key.
    pub(crate) fn register_constructor_by_name(
        &mut self,
        class_symbol: SymbolId,
        func_id: IrFunctionId,
    ) {
        if let Some(sym_info) = self.symbol_table.get_symbol(class_symbol) {
            if let Some(qual_name) = sym_info
                .qualified_name
                .and_then(|q| self.string_interner.get(q))
            {
                self.constructor_name_map
                    .insert(qual_name.to_string(), func_id);
            } else if let Some(name) = self.string_interner.get(sym_info.name) {
                self.constructor_name_map.insert(name.to_string(), func_id);
            }
            if let Some(bare) = self.string_interner.get(sym_info.name) {
                self.constructor_owner_map.insert(func_id, bare.to_string());
            }
        }
    }

    /// Register a forward reference to a stdlib MIR function that will be provided by module merging
    ///
    /// Unlike extern functions (which use C calling convention and are resolved by Cranelift),
    /// stdlib MIR functions use Haxe calling convention and are colocated functions that will
    /// be provided when the stdlib MIR module is merged.
    pub(crate) fn register_stdlib_mir_forward_ref(
        &mut self,
        name: &str,
        mut param_types: Vec<IrType>,
        mut return_type: IrType,
    ) -> IrFunctionId {
        // A bodyless `extern class` method (`Tensor.addInto`) is registered first by
        // `register_function_signature` with `kind` defaulting to `UserDefined`.
        // Reusing that stub as-is would keep the wrong tag, and `own_func_ids`
        // (compilation.rs) then protects it from the stdlib merge's by-name
        // replacement, so the empty stub reaches codegen and traps instead of being
        // replaced by the real body. Retag here, the one place that knows the name is
        // a stdlib MIR wrapper, whichever pass registered the placeholder.
        for (func_id, func) in self.builder.module.functions.iter_mut() {
            if func.name == name {
                func.kind = FunctionKind::MirWrapper;
                return *func_id;
            }
        }

        // A known MIR wrapper's registered signature overrides the inferred one.
        if let Some((correct_params, correct_return)) = self.get_stdlib_mir_wrapper_signature(name)
        {
            debug!(
                "Using registered signature for {}: {} params -> {:?}",
                name,
                correct_params.len(),
                correct_return
            );
            param_types = correct_params;
            return_type = correct_return;
        }

        // Forward reference; the body arrives with the stdlib merge.
        let func_id = IrFunctionId(self.builder.module.next_function_id);
        self.builder.module.next_function_id += 1;

        let params = param_types
            .into_iter()
            .enumerate()
            .map(|(i, ty)| IrParameter {
                name: format!("arg{}", i),
                ty: ty.clone(),
                reg: IrId::new(i as u32),
                by_ref: false,
            })
            .collect();

        // Stdlib MIR wrappers use the C convention (no env param), matching their
        // definitions in thread.rs, channel.rs and sync.rs.
        let signature = IrFunctionSignature {
            parameters: params,
            return_type: return_type.clone(),
            calling_convention: CallingConvention::C,
            can_throw: false,
            type_params: vec![],
            uses_sret: matches!(return_type, IrType::Struct { .. }),
        };

        use crate::ir::{FunctionAttributes, InlineHint, IrControlFlowGraph, Linkage};
        use crate::tast::SymbolId;

        // Empty blocks mark this as a forward declaration.
        let mut attributes = FunctionAttributes::default();
        attributes.linkage = Linkage::Public;
        attributes.inline = InlineHint::Auto;

        let function = IrFunction {
            id: func_id,
            symbol_id: SymbolId::from_raw(0),
            name: name.to_string(),
            // Mirror the bare extern name into `qualified_name` so the stub joins both
            // `func_id_to_qualified_name` and `bare_name_to_id_set` under the same
            // string `stdlib_function_name_map` uses; without it the stub is absent
            // from the qname reverse index and the per-CallDirect rewrite cannot
            // redirect a stale stub id to the real body. Consumers that read
            // `qualified_name.is_some()` as "has a body" also gate on
            // `!cfg.blocks.is_empty()`, so empty stubs stay skipped.
            qualified_name: Some(name.to_string()),
            signature,
            cfg: IrControlFlowGraph::new(), // Empty - will be replaced during merge
            locals: BTreeMap::new(),
            register_types: BTreeMap::new(),
            attributes,
            kind: FunctionKind::MirWrapper,
            source_location: IrSourceLocation::unknown(),
            next_reg_id: 0,
            type_param_tag_fixups: Vec::new(),
            wasm_export: false,
            js_import: None,
        };

        self.builder.module.functions.insert(func_id, function);
        func_id
    }
}
