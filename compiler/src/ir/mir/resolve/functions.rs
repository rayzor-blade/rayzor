//! Resolves call targets to IR function ids, registering externs on demand.

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
    /// Look up a function ID by symbol, checking both local and external function maps
    pub(crate) fn get_function_id(&self, symbol: &SymbolId) -> Option<IrFunctionId> {
        let result = self
            .function_map
            .get(symbol)
            .copied()
            .or_else(|| self.external_function_map.get(symbol).copied());
        result
    }

    /// Resolve a function ID by symbol with a qualified-name fallback across local/external maps.
    pub(crate) fn resolve_function_id_with_qualified_fallback(
        &self,
        symbol: SymbolId,
    ) -> Option<IrFunctionId> {
        if let Some(func_id) = self.get_function_id(&symbol) {
            return Some(func_id);
        }

        let qual_name = self
            .symbol_table
            .get_symbol(symbol)
            .and_then(|s| s.qualified_name)
            .and_then(|qn| self.string_interner.get(qn))?;

        for (local_sym, &local_func_id) in &self.function_map {
            let local_qual = self
                .symbol_table
                .get_symbol(*local_sym)
                .and_then(|s| s.qualified_name)
                .and_then(|qn| self.string_interner.get(qn));
            if local_qual == Some(qual_name) {
                return Some(local_func_id);
            }
        }

        for (ext_sym, &ext_func_id) in &self.external_function_map {
            let ext_qual = self
                .symbol_table
                .get_symbol(*ext_sym)
                .and_then(|s| s.qualified_name)
                .and_then(|qn| self.string_interner.get(qn));
            if ext_qual == Some(qual_name) {
                return Some(ext_func_id);
            }
        }

        self.external_function_name_map.get(qual_name).copied()
    }

    /// Resolve an instance method function ID from receiver type + method name.
    pub(crate) fn resolve_method_function_id(
        &self,
        receiver_type: TypeId,
        method_name: InternedString,
    ) -> Option<IrFunctionId> {
        // Handle Placeholder types (unresolved cross-module types like "Point2D").
        // Construct "ClassName.methodName" and search external_function_name_map directly.
        {
            let type_table = self.type_table;
            if let Some(ti) = type_table.get(receiver_type) {
                if let crate::tast::core::TypeKind::Placeholder { name } = &ti.kind {
                    if let Some(class_name) = self.string_interner.get(*name) {
                        if let Some(method_name_str) = self.string_interner.get(method_name) {
                            // Try direct match first (e.g., "Point2D.add")
                            let qn = format!("{}.{}", class_name, method_name_str);
                            if let Some(&func_id) = self.external_function_name_map.get(&qn) {
                                return Some(func_id);
                            }
                            // Try suffix match for package-qualified names (e.g., "sim.Point2D.add")
                            let suffix = format!(".{}", qn);
                            for (map_name, &func_id) in &self.external_function_name_map {
                                if map_name.ends_with(&suffix) {
                                    return Some(func_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        let class_symbol = self.resolve_receiver_class_symbol(receiver_type)?;
        let method_symbol_opt = self.resolve_class_method_symbol(class_symbol, method_name);
        if let Some(method_symbol) = method_symbol_opt {
            if let Some(func_id) = self.resolve_function_id_with_qualified_fallback(method_symbol) {
                return Some(func_id);
            }
        }

        let class_name_hint = self
            .symbol_table
            .get_symbol(class_symbol)
            .and_then(|s| s.qualified_name.or(Some(s.name)))
            .and_then(|n| self.string_interner.get(n));
        let method_decl_loc = method_symbol_opt
            .and_then(|method_symbol| self.symbol_table.get_symbol(method_symbol))
            .map(|s| s.definition_location)
            .unwrap_or(SourceLocation::unknown());

        // Reject a candidate whose qualified name POSITIVELY names a class
        // other than the receiver's before it can enter the lone-name pool —
        // `Linear.fromQuant` must never bind to `nue.Embedding.fromQuant`
        // just because Linear's body hasn't compiled yet. (The lone-name
        // guard `func_not_other_class` can't see qnames of ids that live in
        // other modules; the SYMBOL qname is always available here.)
        let class_short_for_reject = class_name_hint.and_then(|n| n.rsplit('.').next());
        let sym_other_class = |this: &Self, sym: SymbolId| -> bool {
            let (Some(short), Some(qn)) = (
                class_short_for_reject,
                this.symbol_table
                    .get_symbol(sym)
                    .and_then(|s| s.qualified_name)
                    .and_then(|q| this.string_interner.get(q)),
            ) else {
                return false;
            };
            let parts: Vec<&str> = qn.split('.').collect();
            parts.len() >= 2 && parts[parts.len() - 2] != short
        };

        let mut name_matches = BTreeSet::new();

        for (candidate_sym, &func_id) in &self.function_map {
            if let Some(sym_info) = self.symbol_table.get_symbol(*candidate_sym) {
                if sym_info.name != method_name {
                    continue;
                }
                if !sym_other_class(self, *candidate_sym) {
                    name_matches.insert(func_id);
                }
                if self.symbol_belongs_to_class_hierarchy(*candidate_sym, class_symbol) {
                    return Some(func_id);
                }
                if method_decl_loc.is_valid() && sym_info.definition_location == method_decl_loc {
                    return Some(func_id);
                }
                if let Some(class_hint) = class_name_hint {
                    let qual_match = sym_info
                        .qualified_name
                        .and_then(|qn| self.string_interner.get(qn))
                        .map(|qn| qn.contains(class_hint))
                        .unwrap_or(false);
                    if qual_match {
                        return Some(func_id);
                    }
                }
            }
        }

        for (candidate_sym, &func_id) in &self.external_function_map {
            if let Some(sym_info) = self.symbol_table.get_symbol(*candidate_sym) {
                if sym_info.name != method_name {
                    continue;
                }
                if !sym_other_class(self, *candidate_sym) {
                    name_matches.insert(func_id);
                }
                if self.symbol_belongs_to_class_hierarchy(*candidate_sym, class_symbol) {
                    return Some(func_id);
                }
                if method_decl_loc.is_valid() && sym_info.definition_location == method_decl_loc {
                    return Some(func_id);
                }
                if let Some(class_hint) = class_name_hint {
                    let qual_match = sym_info
                        .qualified_name
                        .and_then(|qn| self.string_interner.get(qn))
                        .map(|qn| qn.contains(class_hint))
                        .unwrap_or(false);
                    if qual_match {
                        return Some(func_id);
                    }
                }
            }
        }

        let method_name_str = self.string_interner.get(method_name)?;
        let class_short_hint = class_name_hint
            .and_then(|n| n.rsplit('.').next())
            .filter(|s| !s.is_empty());
        let mut module_name_matches = BTreeSet::new();
        for (func_id, func) in &self.builder.module.functions {
            if func.name != method_name_str {
                continue;
            }
            module_name_matches.insert(*func_id);
            if let Some(class_hint) = class_name_hint {
                let qual_match = func
                    .qualified_name
                    .as_deref()
                    .map(|qn| {
                        qn.contains(class_hint)
                            || class_short_hint
                                .map(|short| qn.contains(short))
                                .unwrap_or(false)
                    })
                    .unwrap_or(false);
                if qual_match {
                    return Some(*func_id);
                }
            }
        }
        // The bare-name fallbacks below return a lone same-named function. Reject
        // a match that POSITIVELY belongs to a different class than the receiver,
        // so e.g. `NativeStackTrace.callStack` (an extern with a runtime mapping)
        // does NOT bind to `haxe.CallStack.callStack` — which would then call
        // `NativeStackTrace.callStack()` → itself → infinite recursion. Returning
        // None here lets the caller fall through to the runtime-mapping dispatch.
        // Conservative: only rejects when the qname clearly names another class.
        let lone_ok = |this: &Self, fid: IrFunctionId| -> bool {
            class_short_hint.map_or(true, |s| this.func_not_other_class(fid, s))
        };
        if module_name_matches.len() == 1 {
            let fid = *module_name_matches.iter().next().unwrap();
            if lone_ok(self, fid) {
                return Some(fid);
            }
        }

        let mut extern_name_matches = BTreeSet::new();
        for (func_id, func) in &self.builder.module.extern_functions {
            if func.name != method_name_str {
                continue;
            }
            extern_name_matches.insert(*func_id);
        }
        if extern_name_matches.len() == 1 {
            let fid = *extern_name_matches.iter().next().unwrap();
            if lone_ok(self, fid) {
                return Some(fid);
            }
        }

        if name_matches.len() == 1 {
            let fid = *name_matches.iter().next().unwrap();
            if lone_ok(self, fid) {
                return Some(fid);
            }
        }

        // Final fallback: construct "ClassName.methodName" and search external_function_name_map.
        // This handles cross-module method calls where SymbolIds don't match across
        // compilation contexts but the qualified function name is stable.
        if let Some(class_hint) = class_name_hint {
            // Try "ClassName.methodName" (e.g., "Point2D.add")
            let constructed_qn = format!("{}.{}", class_hint, method_name_str);
            if let Some(&func_id) = self.external_function_name_map.get(&constructed_qn) {
                return Some(func_id);
            }
            // Try with just the short class name if class_hint has a package prefix
            if let Some(short) = class_short_hint {
                let short_qn = format!("{}.{}", short, method_name_str);
                if let Some(&func_id) = self.external_function_name_map.get(&short_qn) {
                    return Some(func_id);
                }
            }
        }

        None
    }

    /// Find a MIR function by its bare name in function_map
    pub(crate) fn find_function_by_name(&self, name: &str) -> Option<IrFunctionId> {
        let module = &self.builder.module;
        for (func_id, func) in &module.functions {
            if func.name == name {
                return Some(*func_id);
            }
        }
        None
    }

    /// Find a MIR function by its bare name in external_function_map/extern_functions
    pub(crate) fn find_external_function_by_name(&self, name: &str) -> Option<IrFunctionId> {
        self.external_function_name_map
            .get(name)
            .copied()
            .or_else(|| {
                self.builder
                    .module
                    .extern_functions
                    .iter()
                    .find(|(_, f)| f.name == name)
                    .map(|(id, _)| *id)
            })
    }

    /// Get or register an external runtime function, returning its ID
    ///
    /// This allows calling external runtime functions (like haxe_math_abs) from MIR
    pub(crate) fn get_or_register_extern_function(
        &mut self,
        name: &str,
        mut param_types: Vec<IrType>,
        mut return_type: IrType,
    ) -> IrFunctionId {
        // Override with correct signature if this is a known extern function
        // This is critical for Math functions to get f64 types instead of inferred i64
        if let Some((correct_params, correct_return)) = self.get_extern_function_signature(name) {
            param_types = correct_params;
            return_type = correct_return;
        }

        // FIRST: Check if a MIR wrapper with this name already exists (has blocks)
        // If it does, return that instead of creating an extern
        let existing_mir_wrapper: Option<IrFunctionId> = self
            .builder
            .module
            .functions
            .iter()
            .find(|(_, f)| f.name == name && !f.cfg.blocks.is_empty())
            .map(|(id, _)| *id);

        if let Some(func_id) = existing_mir_wrapper {
            return func_id;
        }

        // SECOND: Check if this name refers to a known stdlib MIR
        // wrapper (a function with a body that will be provided when
        // the stdlib MIR module is merged). Registering such names as
        // Cranelift Import causes the backend to later refuse to
        // define the body with "Invalid to define identifier declared
        // as an import".
        //
        // Detection: either the stdlib runtime mapping flags it
        // explicitly (preferred) OR the name follows the well-known
        // MIR wrapper naming pattern. Both detect the same set; the
        // pattern fallback covers stale mapping entries that predate
        // the `mir_wrapper` macro variant (e.g. `array_length`,
        // `array_push`, `array_index_of`).
        let is_stdlib_wrapper_name = self
            .stdlib_mapping
            .find_by_runtime_name(name)
            .map(|m| m.is_mir_wrapper)
            .unwrap_or(false)
            || name.starts_with("array_")
            || name.starts_with("String_");
        if is_stdlib_wrapper_name {
            return self.register_stdlib_mir_forward_ref(name, param_types, return_type);
        }

        // Check if we already registered this extern function
        // First, find if it exists and collect info (can't mutate while iterating)
        let existing_func: Option<(IrFunctionId, usize)> = self
            .builder
            .module
            .extern_functions
            .iter()
            .find(|(_, ef)| ef.name == name)
            .map(|(id, ef)| (*id, ef.signature.parameters.len()));

        if let Some((func_id, existing_param_count)) = existing_func {
            let new_param_count = param_types.len();

            // If new signature has MORE parameters, update the existing function
            if new_param_count > existing_param_count {
                debug!(
                    "Updating extern '{}' signature: {} params -> {} params",
                    name, existing_param_count, new_param_count
                );

                // Create updated parameters
                let params = param_types
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| IrParameter {
                        name: format!("arg{}", i),
                        ty: ty.clone(),
                        reg: IrId::new(i as u32),
                        by_ref: false,
                    })
                    .collect();

                let new_signature = IrFunctionSignature {
                    parameters: params,
                    return_type: return_type.clone(),
                    calling_convention: CallingConvention::C,
                    can_throw: false,
                    type_params: vec![],
                    uses_sret: false,
                };

                // Update in extern_functions map
                if let Some(ext_func) = self.builder.module.extern_functions.get_mut(&func_id) {
                    ext_func.signature = new_signature.clone();
                }

                // Update in functions map
                if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                    func.signature = new_signature;
                }
            }

            return func_id;
        }

        // Also check in functions (extern functions have empty blocks)
        let existing_func: Option<(IrFunctionId, usize)> = self
            .builder
            .module
            .functions
            .iter()
            .filter(|(_, f)| f.name == name && f.cfg.blocks.is_empty())
            .map(|(id, f)| (*id, f.signature.parameters.len()))
            .next();

        if let Some((func_id, existing_param_count)) = existing_func {
            let new_param_count = param_types.len();

            if new_param_count > existing_param_count {
                debug!(
                    "Updating function '{}' signature: {} params -> {} params",
                    name, existing_param_count, new_param_count
                );

                // Create updated parameters
                let params = param_types
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| IrParameter {
                        name: format!("arg{}", i),
                        ty: ty.clone(),
                        reg: IrId::new(i as u32),
                        by_ref: false,
                    })
                    .collect();

                let new_signature = IrFunctionSignature {
                    parameters: params,
                    return_type: return_type.clone(),
                    calling_convention: CallingConvention::C,
                    can_throw: false,
                    type_params: vec![],
                    uses_sret: false,
                };

                // Update in functions map
                if let Some(f) = self.builder.module.functions.get_mut(&func_id) {
                    f.signature = new_signature.clone();
                }

                // Also update in extern_functions if present
                if let Some(ext_func) = self.builder.module.extern_functions.get_mut(&func_id) {
                    ext_func.signature = new_signature;
                }
            }

            return func_id;
        }

        // println!("  ℹ️  Registering extern function: {}", name);

        // Create new extern function entry
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

        let signature = IrFunctionSignature {
            parameters: params,
            return_type: return_type.clone(),
            calling_convention: CallingConvention::C, // External functions use C calling convention
            can_throw: false,
            type_params: vec![],
            uses_sret: false, // No struct return for C functions
        };

        // Create the IrFunction with empty blocks (extern marker)
        let func = crate::ir::functions::IrFunction {
            id: func_id,
            symbol_id: SymbolId::from_raw(0),
            name: name.to_string(),
            qualified_name: None,
            signature: signature.clone(),
            cfg: crate::ir::blocks::IrControlFlowGraph {
                blocks: std::collections::BTreeMap::new(), // Empty blocks = extern
                entry_block: IrBlockId(0),
                next_block_id: 0,
            },
            locals: std::collections::BTreeMap::new(),
            register_types: std::collections::BTreeMap::new(),
            attributes: crate::ir::functions::FunctionAttributes::default(),
            kind: crate::ir::functions::FunctionKind::ExternC, // Extern C function
            source_location: IrSourceLocation::unknown(),
            next_reg_id: 0,
            type_param_tag_fixups: Vec::new(),
            wasm_export: false,
            js_import: None,
        };

        // Add to both functions and extern_functions maps
        self.builder.module.functions.insert(func_id, func);

        let extern_func = crate::ir::modules::IrExternFunction {
            id: func_id,
            name: name.to_string(),
            symbol_id: SymbolId::from_raw(0), // Placeholder
            signature,
            source: "rayzor_runtime".to_string(),
        };

        self.builder
            .module
            .extern_functions
            .insert(func_id, extern_func);

        // eprintln!(
        //     "  ℹ️  After registration: module has {} functions, {} extern_functions",
        //     self.builder.module.functions.len(),
        //     self.builder.module.extern_functions.len()
        // );

        func_id
    }

    /// Resolve function parameter/return TypeIds from a function-like TypeId.
    /// Follows type aliases until a concrete `TypeKind::Function` is found.
    pub(crate) fn resolve_function_type_signature(
        &self,
        type_id: TypeId,
    ) -> Option<(Vec<TypeId>, TypeId)> {
        use crate::tast::TypeKind;

        let type_table = self.type_table;
        let mut current = type_id;
        let mut visited = BTreeSet::new();
        loop {
            if !visited.insert(current) {
                // Alias cycle detected.
                return None;
            }
            let type_info = type_table.get(current)?;
            match &type_info.kind {
                TypeKind::Function {
                    params,
                    return_type,
                    ..
                } => return Some((params.clone(), *return_type)),
                TypeKind::TypeAlias { target_type, .. } => current = *target_type,
                _ => return None,
            }
        }
    }
}
