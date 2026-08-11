//! Synthetic statics on native struct classes: `@:gpuStruct` layout queries and `cdef`.

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
    pub(crate) fn try_native_struct_static_call(
        &mut self,
        expr: &HirExpr,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call { callee, .. } = &expr.kind else {
            unreachable!("try_native_struct_static_call on a non-Call expression")
        };
        if let HirExprKind::Variable { symbol, .. } = &callee.kind {
            let callee_name = self
                .symbol_table
                .get_symbol(*symbol)
                .and_then(|s| self.string_interner.get(s.name))
                .map(|s| s.to_string());
            // @:gpuStruct synthetic static methods: gpuDef/gpuSize/gpuAlignment
            if matches!(
                callee_name.as_deref(),
                Some("gpuDef")
                    | Some("gpuSize")
                    | Some("gpuAlignment")
                    | Some("gpuVertexLayout")
                    | Some("wgsl")
            ) {
                for (tid, decl) in self.current_hir_types.iter() {
                    if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                        let sym_flags = self
                            .symbol_table
                            .get_symbol(c.symbol_id)
                            .map(|s| s.flags)
                            .unwrap_or(SymbolFlags::NONE);
                        let is_gpu_struct = sym_flags.is_gpu_struct();
                        let is_shader = sym_flags.is_shader();
                        if !is_gpu_struct && !is_shader {
                            continue;
                        }
                        // @:shader wgsl() — handle before has_method check
                        // (synthetic wgsl() may not be in HIR methods list)
                        if is_shader && callee_name.as_deref() == Some("wgsl") {
                            let type_table = self.type_table;
                            match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                c,
                                self.symbol_table,
                                type_table,
                                self.string_interner,
                                self.current_hir_types,
                            ) {
                                Ok(wgsl_source) => {
                                    return self.builder.build_const(IrValue::String(wgsl_source));
                                }
                                Err(e) => {
                                    return self.builder.build_const(IrValue::String(format!(
                                        "/* WGSL error: {} */",
                                        e
                                    )));
                                }
                            }
                        }
                        let has_method = c.methods.iter().any(|m| m.function.symbol_id == *symbol);
                        if has_method {
                            // Find canonical TypeId
                            let canonical_tid = {
                                let type_table = self.type_table;
                                type_table.get(*tid).and_then(|_| Some(*tid)).or_else(|| {
                                    type_table.iter().find_map(|(_, t)| {
                                        if let crate::tast::core::TypeKind::Class {
                                            symbol_id: sid,
                                            ..
                                        } = &t.kind
                                        {
                                            if *sid == c.symbol_id {
                                                return Some(t.id);
                                            }
                                        }
                                        None
                                    })
                                })
                            };
                            // Handle wgsl() on @:shader classes BEFORE layout check
                            if is_shader && callee_name.as_deref() == Some("wgsl") {
                                let type_table = self.type_table;
                                match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                    c,
                                    self.symbol_table,
                                    type_table,
                                    self.string_interner,
                                    self.current_hir_types,
                                ) {
                                    Ok(wgsl_source) => {
                                        return self
                                            .builder
                                            .build_const(IrValue::String(wgsl_source));
                                    }
                                    Err(e) => {
                                        return self.builder.build_const(IrValue::String(format!(
                                            "/* WGSL error: {} */",
                                            e
                                        )));
                                    }
                                }
                            }

                            if let Some(real_tid) = canonical_tid {
                                if let Some(layout) =
                                    self.get_or_compute_gpu_struct_layout(real_tid)
                                {
                                    match callee_name.as_deref().unwrap() {
                                        "gpuDef" => {
                                            let mut full = String::new();
                                            for dep in &layout.dep_typedefs {
                                                full.push_str(dep);
                                            }
                                            full.push_str(&layout.msl_typedef);
                                            return self.builder.build_const(IrValue::String(full));
                                        }
                                        "gpuSize" => {
                                            return self.builder.build_const(IrValue::I32(
                                                layout.total_size as i32,
                                            ));
                                        }
                                        "gpuAlignment" => {
                                            return self.builder.build_const(IrValue::I32(
                                                layout.alignment as i32,
                                            ));
                                        }
                                        "gpuVertexLayout" => {
                                            // Return "stride:offset1,fmt1,loc1;offset2,fmt2,loc2;..."
                                            // Parsed by pure Haxe VertexLayout class
                                            let mut parts = Vec::new();
                                            parts.push(format!("{}", layout.total_size));
                                            for (i, f) in layout.fields.iter().enumerate() {
                                                parts.push(format!(
                                                    "{},{},{}",
                                                    f.byte_offset, f.vertex_format, i
                                                ));
                                            }
                                            let encoded = parts.join(";");
                                            return self
                                                .builder
                                                .build_const(IrValue::String(encoded));
                                        }
                                        "wgsl" => {
                                            // @:shader class — transpile HIR to WGSL
                                            let type_table = self.type_table;
                                            match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                                c,
                                                self.symbol_table,
                                                type_table,
                                                self.string_interner,
                                                self.current_hir_types,
                                            ) {
                                                Ok(wgsl_source) => {
                                                    return self.builder.build_const(
                                                        IrValue::String(wgsl_source),
                                                    );
                                                }
                                                Err(e) => {
                                                    return self.builder.build_const(
                                                        IrValue::String(format!("/* WGSL error: {} */", e)),
                                                    );
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if callee_name.as_deref() == Some("cdef") {
                // Find the @:cstruct class that has this cdef method
                for (tid, decl) in self.current_hir_types.iter() {
                    if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                        let is_cstruct = self
                            .symbol_table
                            .get_symbol(c.symbol_id)
                            .map(|s| s.flags.is_cstruct())
                            .unwrap_or(false);
                        if !is_cstruct {
                            continue;
                        }
                        // Check if this class has a method with our cdef symbol
                        let has_cdef = c.methods.iter().any(|m| m.function.symbol_id == *symbol);
                        if has_cdef {
                            // HIR TypeId may not be in type_table — find canonical TypeId by symbol
                            let canonical_tid = {
                                let type_table = self.type_table;
                                type_table.get(*tid).and_then(|_| Some(*tid)).or_else(|| {
                                    // Scan type_table for a Class with matching symbol_id
                                    type_table.iter().find_map(|(_, t)| {
                                        if let crate::tast::core::TypeKind::Class {
                                            symbol_id: sid,
                                            ..
                                        } = &t.kind
                                        {
                                            if *sid == c.symbol_id {
                                                return Some(t.id);
                                            }
                                        }
                                        None
                                    })
                                })
                            };
                            if let Some(real_tid) = canonical_tid {
                                if let Some(layout) = self.get_or_compute_cstruct_layout(real_tid) {
                                    return self
                                        .builder
                                        .build_const(IrValue::String(layout.cdef_string));
                                }
                            }
                        }
                    }
                }
            }
        }
        *fell_through = true;
        None
    }
}
