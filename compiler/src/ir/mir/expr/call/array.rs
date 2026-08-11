//! Array methods that must reach a type-specific runtime entry rather than
//! the generic one.

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
    pub(crate) fn try_array_runtime_call(
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
            unreachable!("try_array_runtime_call on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        let vname = self
            .symbol_table
            .get_symbol(*symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("?");
        // Array<Float>.push on WASM32 (Variable-callee shape, where the
        // receiver is desugared to args[0] and the value to args[1]).
        // The generic `array_push` MIR wrapper takes an I64 value param,
        // but IrType::I64 lowers to a WASM `i32`, so a Float value (the
        // f64 bit-pattern bitcast to i64) loses its high 32 bits and reads
        // back as 0/garbage. Route Float-element pushes through the f64
        // runtime entry — value stays F64 (→ WASM f64, full 8 bytes). This
        // matches the array-literal lowering and is bit-identical on native.
        if vname == "push" && *is_method && args.len() == 2 {
            let elem_is_f64 = {
                let type_table = self.type_table;
                type_table
                    .get(args[0].ty)
                    .and_then(|t| {
                        if let TypeKind::Array { element_type } = &t.kind {
                            Some(*element_type)
                        } else {
                            None
                        }
                    })
                    .map(|et| self.convert_type(et) == IrType::F64)
                    .unwrap_or(false)
            };
            if elem_is_f64 {
                if let (Some(arr_reg), Some(val_reg)) = (
                    self.lower_expression(&args[0]),
                    self.lower_expression(&args[1]),
                ) {
                    let val_ty = self
                        .builder
                        .get_register_type(val_reg)
                        .unwrap_or(IrType::F64);
                    let val_f64 = if val_ty == IrType::F64 {
                        val_reg
                    } else {
                        self.builder
                            .build_cast(val_reg, val_ty, IrType::F64)
                            .unwrap_or(val_reg)
                    };
                    let push_fn = self.get_or_register_extern_function(
                        "haxe_array_push_f64",
                        vec![IrType::Ptr(Box::new(IrType::I64)), IrType::F64],
                        IrType::Void,
                    );
                    return self.builder.build_call_direct(
                        push_fn,
                        vec![arr_reg, val_f64],
                        IrType::Void,
                    );
                }
            }
        }

        // Array.join: the generic array_join runtime treats every
        // element as a HaxeString pointer, which SIGSEGVs for non-
        // String element types. Route through haxe_array_join_typed
        // with the element's type tag so each element is converted
        // via Std.string first (1=Int 2=Bool 4=Float 5=String 6=Ref).
        if vname == "join" && *is_method && args.len() == 2 {
            let elem_tag: i32 = {
                let type_table = self.type_table;
                type_table
                    .get(args[0].ty)
                    .and_then(|t| {
                        if let TypeKind::Array { element_type } = &t.kind {
                            Some(*element_type)
                        } else {
                            None
                        }
                    })
                    .and_then(|et| type_table.get(et).map(|t| t.kind.clone()))
                    .map(|k| match k {
                        TypeKind::Int => 1,
                        TypeKind::Bool => 2,
                        TypeKind::Float => 4,
                        TypeKind::String => 5,
                        _ => 6,
                    })
                    .unwrap_or(5)
            };
            if let (Some(arr_reg), Some(sep_reg)) = (
                self.lower_expression(&args[0]),
                self.lower_expression(&args[1]),
            ) {
                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                let tag_reg = self.builder.build_const(IrValue::I32(elem_tag))?;
                let join_fn = self.get_or_register_extern_function(
                    "haxe_array_join_typed",
                    vec![ptr_void.clone(), ptr_void.clone(), IrType::I32],
                    ptr_void.clone(),
                );
                return self.builder.build_call_direct(
                    join_fn,
                    vec![arr_reg, sep_reg, tag_reg],
                    ptr_void,
                );
            }
        }
        *fell_through = true;
        None
    }
}
