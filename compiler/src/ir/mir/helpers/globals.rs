//! Global initialisers: constant evaluation and per-type defaults.

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
    pub(crate) fn refine_global_type_from_initializer(
        &self,
        ty: IrType,
        initializer: Option<&IrValue>,
    ) -> IrType {
        let is_placeholder_ptr =
            matches!(&ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void));
        if !(is_placeholder_ptr || ty == IrType::Any) {
            return ty;
        }

        match initializer {
            Some(IrValue::Bool(_)) => IrType::Bool,
            Some(IrValue::I8(_)) => IrType::I8,
            Some(IrValue::I16(_)) => IrType::I16,
            Some(IrValue::I32(_)) => IrType::I32,
            Some(IrValue::I64(_)) => IrType::I64,
            Some(IrValue::U8(_)) => IrType::U8,
            Some(IrValue::U16(_)) => IrType::U16,
            Some(IrValue::U32(_)) => IrType::U32,
            Some(IrValue::U64(_)) => IrType::U64,
            Some(IrValue::F32(_)) => IrType::F32,
            Some(IrValue::F64(_)) => IrType::F64,
            _ => ty,
        }
    }

    pub(crate) fn reset_value_for_global(&self, global: &IrGlobal) -> IrValue {
        match global.initializer.as_ref() {
            Some(IrValue::Undef) | None => match &global.ty {
                IrType::Void => IrValue::Void,
                IrType::Bool => IrValue::Bool(false),
                IrType::I8 => IrValue::I8(0),
                IrType::I16 => IrValue::I16(0),
                IrType::I32 => IrValue::I32(0),
                IrType::I64 => IrValue::I64(0),
                IrType::U8 => IrValue::U8(0),
                IrType::U16 => IrValue::U16(0),
                IrType::U32 => IrValue::U32(0),
                IrType::U64 => IrValue::U64(0),
                IrType::F32 => IrValue::F32(0.0),
                IrType::F64 => IrValue::F64(0.0),
                _ => IrValue::Null,
            },
            Some(init) => init.clone(),
        }
    }

    /// Try to evaluate an expression as a constant value for static field initialization
    pub(crate) fn try_evaluate_constant_init(&self, init_expr: &HirExpr) -> Option<IrValue> {
        match &init_expr.kind {
            HirExprKind::Literal(lit) => {
                match lit {
                    HirLiteral::Bool(b) => Some(IrValue::Bool(*b)),
                    HirLiteral::Int(i) => Some(IrValue::I64(*i)),
                    HirLiteral::Float(f) => Some(IrValue::F64(*f)),
                    HirLiteral::String(_) => None, // Strings need special handling
                    HirLiteral::Regex { .. } => None,
                }
            }
            // Constant-fold binary ops on literal operands (e.g., 1 << 16)
            HirExprKind::Binary { op, lhs, rhs } => {
                let l = self.try_evaluate_constant_init(lhs)?;
                let r = self.try_evaluate_constant_init(rhs)?;
                let (li, ri) = match (&l, &r) {
                    (IrValue::I64(a), IrValue::I64(b)) => (*a, *b),
                    (IrValue::I32(a), IrValue::I32(b)) => (*a as i64, *b as i64),
                    (IrValue::I64(a), IrValue::I32(b)) => (*a, *b as i64),
                    (IrValue::I32(a), IrValue::I64(b)) => (*a as i64, *b),
                    _ => return None,
                };
                let result = match op {
                    HirBinaryOp::Add => li.wrapping_add(ri),
                    HirBinaryOp::Sub => li.wrapping_sub(ri),
                    HirBinaryOp::Mul => li.wrapping_mul(ri),
                    HirBinaryOp::Div if ri != 0 => li / ri,
                    HirBinaryOp::Mod if ri != 0 => li % ri,
                    HirBinaryOp::Shl => li.wrapping_shl(ri as u32),
                    HirBinaryOp::Shr => li.wrapping_shr(ri as u32),
                    HirBinaryOp::BitAnd => li & ri,
                    HirBinaryOp::BitOr => li | ri,
                    HirBinaryOp::BitXor => li ^ ri,
                    _ => return None,
                };
                // Return I32 for values that fit, matching Haxe Int type
                if result >= i32::MIN as i64 && result <= i32::MAX as i64 {
                    Some(IrValue::I32(result as i32))
                } else {
                    Some(IrValue::I64(result))
                }
            }
            HirExprKind::Unary { op, operand } => {
                let v = self.try_evaluate_constant_init(operand)?;
                match (op, &v) {
                    (HirUnaryOp::Neg, IrValue::I64(i)) => Some(IrValue::I64(-i)),
                    (HirUnaryOp::Neg, IrValue::F64(f)) => Some(IrValue::F64(-f)),
                    (HirUnaryOp::BitNot, IrValue::I64(i)) => Some(IrValue::I64(!i)),
                    (HirUnaryOp::Not, IrValue::Bool(b)) => Some(IrValue::Bool(!b)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Build a type-based default value: 0 for Int, 0.0 for Float, false for Bool, "" for String,
    /// recursive default or no-arg constructor for classes, null otherwise.
    pub(crate) fn build_type_default(&mut self, field_type_id: TypeId) -> Option<IrId> {
        let type_kind = {
            let type_table = self.type_table;
            type_table.get(field_type_id).map(|t| t.kind.clone())
        };

        match type_kind.as_ref() {
            Some(TypeKind::Int) => self.builder.build_const(IrValue::I32(0)),
            Some(TypeKind::Float) => self.builder.build_const(IrValue::F64(0.0)),
            Some(TypeKind::Bool) => self.builder.build_const(IrValue::Bool(false)),
            Some(TypeKind::String) => self.builder.build_string(String::new()),
            _ => {
                // Classes default to an instance where one can be built, else null.
                Some(
                    self.try_default_for_class_field(field_type_id)
                        .unwrap_or_else(|| self.builder.build_null().unwrap()),
                )
            }
        }
    }

    /// Try to produce a default value for a class-typed field:
    /// 1. If the field's class has @:derive(Default), recursively default-initialize it
    /// 2. Else if the field's class has a no-arg constructor, call it
    /// 3. Else return None (caller should use null)
    pub(crate) fn try_default_for_class_field(&mut self, field_type_id: TypeId) -> Option<IrId> {
        let (class_symbol, class_name) = {
            let type_table = self.type_table;
            match type_table.get(field_type_id).map(|t| &t.kind) {
                Some(TypeKind::Class { symbol_id, .. }) => {
                    let name = self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|s| self.string_interner.get(s.name))
                        .map(|s| s.to_string());
                    Some((*symbol_id, name))
                }
                _ => None,
            }
        }?;

        if self.derive_default_classes.contains(&class_symbol) {
            return self.lower_derived_default(class_symbol);
        }

        let class_name_str = class_name?;
        let ctor_id = self.constructor_name_map.get(&class_name_str).copied()?;

        let sig_params = self
            .builder
            .module
            .functions
            .get(&ctor_id)
            .map(|f| f.signature.parameters.len())
            .unwrap_or(0);

        // Every param must have a default, or `this` must be the only one.
        let all_optional = if sig_params <= 1 {
            true
        } else if let Some(defaults) = self.function_param_defaults.get(&ctor_id) {
            // defaults[0] is `this`, always None.
            defaults.iter().skip(1).all(|d| d.is_some())
        } else {
            sig_params <= 1
        };

        if !all_optional {
            return None;
        }

        let alloc_size = self
            .class_alloc_sizes_by_name
            .get(&class_name_str)
            .copied()
            .unwrap_or(16);
        let obj_ptr = self.build_heap_alloc(alloc_size)?;

        // Store type_id header
        let type_id_const = self
            .builder
            .build_const(IrValue::I64(class_symbol.as_raw() as i64))?;
        let zero = self.builder.build_const(IrValue::I32(0))?;
        let header_ptr = self.builder.build_gep(obj_ptr, vec![zero], IrType::I64)?;
        self.builder.build_store(header_ptr, type_id_const);

        let mut args = vec![obj_ptr];
        self.fill_default_args(ctor_id, &mut args, true);
        self.builder.build_call_direct(ctor_id, args, IrType::Void);

        self.register_class_hints.insert(obj_ptr, class_name_str);

        Some(obj_ptr)
    }
}
