//! C-struct and GPU-struct classes, and the TCC/JS extern plumbing.

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
    /// Check if a class symbol has the @:cstruct flag
    pub(crate) fn is_cstruct_class(&self, type_id: TypeId) -> bool {
        let type_table = self.type_table;
        if let Some(type_info) = type_table.get(type_id) {
            if let Some(symbol_id) = type_info.symbol_id() {
                if let Some(sym) = self.symbol_table.get_symbol(symbol_id) {
                    return sym.flags.is_cstruct();
                }
            }
        }
        false
    }

    /// Get the C type string for a pointer target type.
    /// If target is @:cstruct, returns "StructName*"; otherwise "void*".
    /// For pointers, we only need the target struct's name, not its full layout,
    /// which avoids infinite recursion for self-referential types like `Ptr<Node>`.
    pub(crate) fn cstruct_ptr_c_type(
        &mut self,
        target_type_id: TypeId,
        dep_cdefs: &mut Vec<String>,
    ) -> String {
        let target_info = {
            let type_table = self.type_table;
            type_table.get(target_type_id).map(|t| t.kind.clone())
        };
        if let Some(TypeKind::Class { symbol_id, .. }) = target_info {
            let is_cstruct = self
                .symbol_table
                .get_symbol(symbol_id)
                .map(|s| s.flags.is_cstruct())
                .unwrap_or(false);
            if is_cstruct {
                // Get the C name without computing full layout (avoids infinite recursion)
                let no_mangle = self
                    .symbol_table
                    .get_symbol(symbol_id)
                    .map(|s| s.flags.is_no_mangle())
                    .unwrap_or(false);
                let class_name = self
                    .symbol_table
                    .get_symbol(symbol_id)
                    .and_then(|s| self.string_interner.get(s.name))
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let c_name = if no_mangle {
                    class_name
                } else {
                    class_name.replace('.', "_")
                };

                if let Some(layout) = self.cstruct_layouts.get(&target_type_id).cloned() {
                    for dep in &layout.dep_cdefs {
                        if !dep_cdefs.contains(dep) {
                            dep_cdefs.push(dep.clone());
                        }
                    }
                    let own_cdef = {
                        let mut s = format!("typedef struct {{ ");
                        for f in &layout.fields {
                            s.push_str(&format!("{} {}; ", f.c_type, f.name));
                        }
                        s.push_str(&format!("}} {};\n", layout.c_name));
                        s
                    };
                    if !dep_cdefs.contains(&own_cdef) {
                        dep_cdefs.push(own_cdef);
                    }
                }
                // A pointer only needs a forward declaration in C, so the name suffices.
                return format!("{}*", c_name);
            }
        }
        "void*".to_string()
    }

    /// Check if a class symbol has the @:gpuStruct flag
    pub(crate) fn is_gpu_struct_class(&self, type_id: TypeId) -> bool {
        let type_table = self.type_table;
        if let Some(type_info) = type_table.get(type_id) {
            if let Some(symbol_id) = type_info.symbol_id() {
                if let Some(sym) = self.symbol_table.get_symbol(symbol_id) {
                    return sym.flags.is_gpu_struct();
                }
            }
        }
        false
    }

    /// Declare a single TCC extern function and register it
    pub(crate) fn declare_tcc_extern(
        &mut self,
        name: &str,
        param_types: Vec<IrType>,
        return_type: IrType,
        dummy_id_offset: u32,
    ) -> IrFunctionId {
        use crate::ir::modules::IrExternFunction;
        use crate::ir::{CallingConvention, IrFunctionSignature, IrId, IrParameter};

        let parameters: Vec<IrParameter> = param_types
            .into_iter()
            .enumerate()
            .map(|(i, ty)| IrParameter {
                name: format!("p{}", i),
                ty,
                reg: IrId::new(0),
                by_ref: false,
            })
            .collect();

        let signature = IrFunctionSignature {
            parameters,
            return_type,
            calling_convention: CallingConvention::C,
            can_throw: false,
            type_params: vec![],
            uses_sret: false,
        };

        let func_id = self.builder.module.alloc_function_id();
        let extern_func = IrExternFunction {
            id: func_id,
            name: name.to_string(),
            symbol_id: SymbolId::from_raw(u32::MAX - 100 - dummy_id_offset),
            signature,
            source: "rayzor_runtime".to_string(),
        };
        self.builder
            .module
            .extern_functions
            .insert(func_id, extern_func);
        func_id
    }

    /// Propagate @:jsImport metadata from symbol/owning class to MIR function.
    pub(crate) fn propagate_js_import(
        &mut self,
        symbol_id: SymbolId,
        func_id: IrFunctionId,
        this_type: Option<TypeId>,
    ) {
        self.propagate_js_import_with_class(symbol_id, func_id, this_type, None);
    }

    pub(crate) fn propagate_js_import_with_class(
        &mut self,
        symbol_id: SymbolId,
        func_id: IrFunctionId,
        this_type: Option<TypeId>,
        class_symbol: Option<SymbolId>,
    ) {
        let mut js_mod: Option<String> = None;
        let mut js_func: Option<String> = None;

        // Resolve the owning class's JS module (for @:jsMethod/@:jsGet/@:jsSet)
        let class_js_module: Option<String> = {
            let csid = class_symbol
                .or(self.current_class_symbol)
                .or_else(|| this_type.and_then(|t| self.class_type_to_symbol.get(&t).copied()));
            csid.and_then(|csid| {
                self.symbol_table.get_symbol(csid).and_then(|class_sym| {
                    class_sym.js_import.and_then(|(mod_is, _)| {
                        self.string_interner.get(mod_is).map(|s| s.to_string())
                    })
                })
            })
        };

        // Check method's own JS metadata (@:jsMethod, @:jsGet, @:jsSet, @:jsFunction)
        if let Some(sym) = self.symbol_table.get_symbol(symbol_id) {
            if let Some((mod_is, func_is)) = sym.js_import {
                let mod_str = self.string_interner.get(mod_is).unwrap_or("").to_string();
                let func_str = self.string_interner.get(func_is).unwrap_or("").to_string();

                if mod_str == "__jsMethod__" {
                    // @:jsMethod/@:jsGet/@:jsSet — inherit module from class's @:jsImport
                    js_mod = class_js_module.clone();
                    js_func = Some(func_str);
                } else {
                    // @:jsFunction or @:jsImport with explicit module
                    js_mod = Some(mod_str);
                    js_func = Some(func_str);
                }
            }
        }

        // If method has no JS metadata, check if the class has @:jsImport
        // and use the method's MIR name as the import name (fallback)
        if js_mod.is_none() {
            if let Some(ref class_mod) = class_js_module {
                js_mod = Some(class_mod.clone());
                if let Some(f) = self.builder.module.functions.get(&func_id) {
                    js_func = Some(f.name.clone());
                }
            }
        }

        if let (Some(module), Some(func_name)) = (js_mod, js_func) {
            if let Some(func_mut) = self.builder.module.functions.get_mut(&func_id) {
                func_mut.js_import = Some((module, func_name));
            }
        }
    }
}
