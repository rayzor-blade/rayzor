//! Enum constructors applied to payload arguments.

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
    pub(crate) fn try_enum_constructor_via_field(
        &mut self,
        expr: &HirExpr,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            ..
        } = &expr.kind
        else {
            unreachable!("try_enum_constructor_via_field on a non-Call expression")
        };
        if let HirExprKind::Field { object, field } = &callee.kind {
            if let HirExprKind::Variable {
                symbol: enum_symbol,
                ..
            } = &object.kind
            {
                if let Some(enum_sym) = self.symbol_table.get_symbol(*enum_symbol) {
                    if enum_sym.kind == crate::tast::SymbolKind::Enum {
                        let field_sym = self.symbol_table.get_symbol(*field);
                        let field_name = field_sym
                            .and_then(|s| self.string_interner.get(s.name))
                            .unwrap_or("");

                        if let Some(variants) = self.symbol_table.get_enum_variants(*enum_symbol) {
                            for (idx, variant_id) in variants.iter().enumerate() {
                                let variant_sym = self.symbol_table.get_symbol(*variant_id);
                                let variant_name = variant_sym
                                    .and_then(|s| self.string_interner.get(s.name))
                                    .unwrap_or("");
                                let id_match = *variant_id == *field;
                                let name_match = !id_match && variant_name == field_name;

                                if id_match || name_match {
                                    let field_count =
                                        self.get_enum_variant_field_count(*enum_symbol, idx);
                                    if field_count == 0 {
                                        if self.enum_is_boxed(*enum_symbol) {
                                            return self.build_boxed_enum_tag_only(
                                                *enum_symbol,
                                                idx as i32,
                                            );
                                        }
                                        return self.builder.build_const(IrValue::I64(idx as i64));
                                    }

                                    let constructor_args = if *is_method
                                        && !args.is_empty()
                                        && self.is_enum_symbol_expr(&args[0])
                                    {
                                        &args[1..]
                                    } else {
                                        args
                                    };
                                    let field_types = self.enum_constructor_field_types(
                                        expr.ty,
                                        *enum_symbol,
                                        variant_sym?.name,
                                    );
                                    return self.build_boxed_enum_with_fields(
                                        *enum_symbol,
                                        idx as i32,
                                        field_count,
                                        constructor_args,
                                        &field_types,
                                    );
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

    pub(crate) fn try_enum_constructor(
        &mut self,
        expr: &HirExpr,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            ..
        } = &expr.kind
        else {
            unreachable!("try_enum_constructor on a non-Call expression")
        };
        if let HirExprKind::Variable { symbol, .. } = &callee.kind {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
                use crate::tast::SymbolKind;
                if sym.kind == SymbolKind::EnumVariant
                    || (sym.kind == SymbolKind::Function
                        && self
                            .symbol_table
                            .find_parent_enum_for_constructor(*symbol)
                            .is_some())
                {
                    if let Some(parent_enum_id) =
                        self.symbol_table.find_parent_enum_for_constructor(*symbol)
                    {
                        if let Some(variants) = self.symbol_table.get_enum_variants(parent_enum_id)
                        {
                            for (idx, variant_id) in variants.iter().enumerate() {
                                if *variant_id == *symbol {
                                    let field_count =
                                        self.get_enum_variant_field_count(parent_enum_id, idx);

                                    if field_count == 0 {
                                        // If enum has parameterized variants, all variants must be boxed
                                        if self.enum_is_boxed(parent_enum_id) {
                                            return self.build_boxed_enum_tag_only(
                                                parent_enum_id,
                                                idx as i32,
                                            );
                                        }
                                        // Pure discriminant enum - return index directly
                                        return self.builder.build_const(IrValue::I64(idx as i64));
                                    }

                                    // Store each parameter at byte offset 8 + i*8
                                    // When is_method=true, args[0] is the enum class reference
                                    // (receiver), not a constructor field. Skip it.
                                    let constructor_args: &[HirExpr] = if *is_method {
                                        if args.len() > 1 { &args[1..] } else { &[] }
                                    } else {
                                        args
                                    };
                                    let field_types = self.enum_constructor_field_types(
                                        expr.ty,
                                        parent_enum_id,
                                        sym.name,
                                    );
                                    return self.build_boxed_enum_with_fields(
                                        parent_enum_id,
                                        idx as i32,
                                        field_count,
                                        constructor_args,
                                        &field_types,
                                    );
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

    fn enum_constructor_field_types(
        &self,
        result_ty: TypeId,
        enum_symbol: SymbolId,
        variant_name: InternedString,
    ) -> Vec<(IrType, TypeId)> {
        let enum_id = self.enum_runtime_id(enum_symbol);
        let concrete_ty = self
            .let_target_type_hint
            .filter(|ty| {
                self.resolve_enum_symbol(*ty)
                    .is_some_and(|sym| self.enum_runtime_id(sym) == enum_id)
            })
            .unwrap_or(result_ty);
        self.get_enum_variant_field_types(concrete_ty, variant_name)
    }
}
