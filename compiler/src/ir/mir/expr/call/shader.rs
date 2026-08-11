//! `@:shader` entry points: a `wgsl()` call compiles the shader body.

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
    pub(crate) fn try_shader_call(
        &mut self,
        expr: &HirExpr,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call { callee, .. } = &expr.kind else {
            unreachable!("try_shader_call on a non-Call expression")
        };
        if let HirExprKind::Field { object, field } = &callee.kind {
            let field_name_check = self
                .symbol_table
                .get_symbol(*field)
                .and_then(|s| self.string_interner.get(s.name));
            let is_wgsl = field_name_check == Some("wgsl");
            if is_wgsl {
                if let HirExprKind::Variable {
                    symbol: class_sym, ..
                } = &object.kind
                {
                    let is_shader = self
                        .symbol_table
                        .get_symbol(*class_sym)
                        .map(|s| s.flags.is_shader())
                        .unwrap_or(false);
                    if is_shader {
                        for (_tid, decl) in self.current_hir_types.iter() {
                            if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                                if c.symbol_id == *class_sym {
                                    let tt = self.type_table;
                                    match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                        c,
                                        self.symbol_table,
                                        tt,
                                        self.string_interner,
                                        self.current_hir_types,
                                    ) {
                                        Ok(wgsl) => {
                                            return self.builder.build_const(IrValue::String(wgsl))
                                        }
                                        Err(e) => {
                                            return self.builder.build_const(IrValue::String(
                                                format!("/* WGSL error: {} */", e),
                                            ))
                                        }
                                    }
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

    pub(crate) fn try_wgsl_method_call(
        &mut self,
        expr: &HirExpr,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call { callee, .. } = &expr.kind else {
            unreachable!("try_wgsl_method_call on a non-Call expression")
        };
        let HirExprKind::Field { object, field } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        let method_name_interned = self.symbol_table.get_symbol(*field).map(|s| s.name);
        let method_name = method_name_interned.and_then(|name| self.string_interner.get(name));
        if method_name == Some("wgsl") {
            // Find the @:shader class from the object type
            let obj_sym = {
                let type_table = self.type_table;
                type_table.get(object.ty).and_then(|t| {
                    if let crate::tast::core::TypeKind::Class { symbol_id, .. } = &t.kind {
                        Some(*symbol_id)
                    } else {
                        None
                    }
                })
            };
            let is_shader = obj_sym
                .and_then(|sid| self.symbol_table.get_symbol(sid))
                .map(|s| s.flags.is_shader())
                .unwrap_or(false);
            if is_shader {
                // Find the HIR class and transpile
                for (_tid, decl) in self.current_hir_types.iter() {
                    if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                        if Some(c.symbol_id) == obj_sym {
                            let type_table = self.type_table;
                            match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                c,
                                self.symbol_table,
                                type_table,
                                self.string_interner,
                                self.current_hir_types,
                            ) {
                                Ok(wgsl) => return self.builder.build_const(IrValue::String(wgsl)),
                                Err(e) => {
                                    return self.builder.build_const(IrValue::String(format!(
                                        "/* WGSL error: {} */",
                                        e
                                    )))
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
