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
        let mut signature = self.build_function_signature(hir_func);

        // For instance methods, add implicit 'this' parameter
        // 'this' is always a pointer to the class instance, regardless of generic parameters
        if let Some(type_id) = this_type {
            let this_type = match self.convert_type(type_id) {
                IrType::Ptr(_) => IrType::Ptr(Box::new(IrType::Void)),
                // If convert_type failed to resolve (e.g., generic class without instantiation),
                // default to pointer since 'this' is always a pointer to instance
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

        // A bodyless declaration (e.g. an `extern class Tensor` method like
        // `addInto`) lands here with no HIR statements to lower, so it's
        // registered as an empty forward-ref placeholder. `IrFunction::new`
        // defaults `kind` to `UserDefined` — correct for a genuine forward
        // reference to a user function, but WRONG when the name is actually
        // a known stdlib MIR-wrapper (e.g. `Tensor_addInto`): `own_func_ids`
        // (compilation.rs) uses `kind != MirWrapper` to decide whether this
        // file's own compile "owns" the function and must PROTECT it from
        // the stdlib merge's by-name replacement. A wrongly-`UserDefined`
        // stdlib-wrapper stub gets protected, survives the merge as
        // permanently empty, and traps (`brk`, TrapCode::user(1)) the first
        // time it's called. A file that declares/references the extern
        // method but never itself CALLS it never reaches the call-site-side
        // fixup in `register_stdlib_mir_forward_ref`, so the tag must be
        // set correctly here too, at creation, via the same name-based
        // stdlib lookup used elsewhere.
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

        // Propagate @:jsImport
        self.propagate_js_import(symbol_id, func_id, this_type);

        self.builder.finish_function(); // Close to allow next function to start
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
        let mut signature =
            self.build_function_signature_with_class_type_params(hir_func, class_type_params);

        // For instance methods, add implicit 'this' parameter
        if let Some(type_id) = this_type {
            // Check the actual HIR TypeKind to distinguish abstracts from classes.
            // convert_type() can misresolve class types to I32 in cross-module
            // compilation when the type table entry is missing. Use TypeKind directly.
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
            // TypeIds are context-local. When the id drifts — adding ONE type to
            // StdTypes shifts every TypeId while SymbolIds stay put — a class's
            // id can land on an abstract-over-Int, and the check above then
            // types `this` BY VALUE. That truncates the 64-bit receiver to i32
            // and every field read off it faults (StringBuf.add/addSub/... ->
            // SIGSEGV in haxe_bytes_data_address). The HIR module we are
            // lowering is authoritative about what it declares, so let it veto:
            // if HIR says this id is a CLASS, it is a class, whatever the
            // (possibly stale) type_table entry says.
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

        // Propagate @:jsImport (same as register_function_signature)
        self.propagate_js_import(symbol_id, func_id, this_type);

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
        // Constructor signature: takes implicit 'this' parameter + constructor params, returns void
        // 'this' is always a pointer to the class instance, regardless of generic parameters
        let this_type = match self.convert_type(type_id) {
            IrType::Ptr(_) => IrType::Ptr(Box::new(IrType::Void)),
            _ => IrType::Ptr(Box::new(IrType::Void)),
        };
        let mut sig_builder = FunctionSignatureBuilder::new().param("this".to_string(), this_type);

        // Add constructor parameters
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

        // Assign register IDs to parameters
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

        // Also register with TypeId derived from class SymbolId as a fallback
        let fallback_type_id = TypeId::from_raw(class_symbol.as_raw());
        if fallback_type_id != type_id {
            self.constructor_map.insert(fallback_type_id, func_id);
        }

        self.builder.finish_function(); // Close the stub
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

        // Add class type parameters
        for type_param in class_type_params {
            let param_name = self
                .string_interner
                .get(type_param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("T{}", type_param.name.as_raw()));
            sig_builder = sig_builder.type_param(param_name);
        }

        // Add constructor parameters
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
            // eprintln!("[CTOR_TP] {}.new param={} ty={:?} kind={:?} → {:?}",
            //     self.string_interner.get(self.symbol_table.get_symbol(class_symbol).unwrap().name).unwrap_or("?"),
            //     param_name, param.ty, kind, ir_ty);
            sig_builder = sig_builder.param(param_name, ir_ty);
        }

        let mut signature = sig_builder.returns(IrType::Void).build();

        // Assign register IDs
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

        self.builder.finish_function();
    }

    /// Register a constructor by qualified name for cross-file resolution
    /// This is critical when the TypeId differs between files (e.g., loading StringIteratorUnicode
    /// as a dependency gives it a different TypeId than when StringTools.hx references it)
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
                // Fallback to simple name if no qualified name
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
        // Check if already registered. A bodyless `extern class` method
        // declaration (e.g. `Tensor.addInto`) is registered FIRST by
        // `register_function_signature`, which defaults new IrFunctions to
        // `FunctionKind::UserDefined` (functions.rs IrFunction::new) — it
        // has no concept of "this is a stdlib MIR-wrapper name". When a
        // call site THEN resolves the same name through here, the empty
        // stub is reused as-is, still tagged UserDefined. That tag makes
        // `own_func_ids` (compilation.rs) treat it as genuine user code and
        // PROTECT it from the stdlib merge's by-name replacement — so the
        // permanently-empty stub survives to codegen and traps at runtime
        // (`brk`, TrapCode::user(1)) the first time it's called, instead of
        // being replaced by the real body (e.g. `build_tensor_add_into`).
        // Retagging here, at the one place that KNOWS this name is a
        // stdlib MIR wrapper, closes the gap regardless of which pass
        // registered the placeholder first.
        for (func_id, func) in self.builder.module.functions.iter_mut() {
            if func.name == name {
                func.kind = FunctionKind::MirWrapper;
                return *func_id;
            }
        }

        // Override with correct signature if this is a known MIR wrapper
        // This fixes the bug where Thread_spawn gets wrong signature from inferred lambda type
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

        // Create forward reference with Haxe calling convention (will be replaced during merge)
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

        // Stdlib MIR wrappers use C calling convention (no env param)
        // This matches the actual definitions in thread.rs, channel.rs, sync.rs
        let signature = IrFunctionSignature {
            parameters: params,
            return_type: return_type.clone(),
            calling_convention: CallingConvention::C, // C calling convention for stdlib MIR wrappers
            can_throw: false,
            type_params: vec![],
            uses_sret: matches!(return_type, IrType::Struct { .. }),
        };

        use crate::ir::{FunctionAttributes, InlineHint, IrControlFlowGraph, Linkage};
        use crate::tast::SymbolId;

        // Create stub function (empty blocks = forward declaration)
        let mut attributes = FunctionAttributes::default();
        attributes.linkage = Linkage::Public;
        attributes.inline = InlineHint::Auto;

        let function = IrFunction {
            id: func_id,
            symbol_id: SymbolId::from_raw(0),
            name: name.to_string(),
            // Mirror the bare extern name into `qualified_name` so the stub
            // joins both `func_id_to_qualified_name` and
            // `bare_name_to_id_set` keyed by the same string used in
            // `stdlib_function_name_map`. Without this, stubs registered
            // by this helper are silently absent from the qname reverse
            // index (see compilation.rs:3682) and the per-CallDirect
            // rewrite cannot redirect a stale stub id to the real body.
            // Downstream code that treats `qualified_name.is_some()` as
            // "real body" already gates on `!cfg.blocks.is_empty()` too,
            // so empty stubs remain safely skipped for serialization and
            // early-rename paths.
            qualified_name: Some(name.to_string()),
            signature,
            cfg: IrControlFlowGraph::new(), // Empty - will be replaced during merge
            locals: BTreeMap::new(),
            register_types: BTreeMap::new(),
            attributes,
            kind: FunctionKind::MirWrapper, // Stdlib MIR function
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
