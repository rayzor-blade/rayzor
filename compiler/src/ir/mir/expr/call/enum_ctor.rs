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
                                            return self.build_boxed_enum_tag_only(idx as i32);
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
                                    return self.build_boxed_enum_with_fields(
                                        idx as i32,
                                        field_count,
                                        constructor_args,
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
                    // Find the parent enum and variant index
                    if let Some(parent_enum_id) =
                        self.symbol_table.find_parent_enum_for_constructor(*symbol)
                    {
                        if let Some(variants) = self.symbol_table.get_enum_variants(parent_enum_id)
                        {
                            for (idx, variant_id) in variants.iter().enumerate() {
                                if *variant_id == *symbol {
                                    // Get variant field count from HIR
                                    let field_count =
                                        self.get_enum_variant_field_count(parent_enum_id, idx);

                                    if field_count == 0 {
                                        // If enum has parameterized variants, all variants must be boxed
                                        if self.enum_is_boxed(parent_enum_id) {
                                            return self.build_boxed_enum_tag_only(idx as i32);
                                        }
                                        // Pure discriminant enum - return index directly
                                        return self.builder.build_const(IrValue::I64(idx as i64));
                                    }

                                    // Has parameters - allocate boxed enum struct
                                    // Layout: [tag:i32][pad:i32][field0:i64][field1:i64]...
                                    let struct_size = 8 + 8 * field_count; // 8 for tag+pad, 8 per field

                                    // Allocate memory
                                    let size_const = self
                                        .builder
                                        .build_const(IrValue::I64(struct_size as i64))?;
                                    let alloc_func = self.get_or_register_extern_function(
                                        "malloc",
                                        vec![IrType::I64],
                                        IrType::Ptr(Box::new(IrType::I8)),
                                    );
                                    let ptr = self.builder.build_call_direct(
                                        alloc_func,
                                        vec![size_const],
                                        IrType::Ptr(Box::new(IrType::I8)),
                                    )?;

                                    // Store tag at offset 0 (as i32)
                                    // Note: GEP multiplies index by element size, so we use I8 elements
                                    // for byte-based addressing, then bitcast to the target type
                                    let zero_offset = self.builder.build_const(IrValue::I64(0))?;
                                    let tag_ptr = self.builder.build_gep(
                                        ptr,
                                        vec![zero_offset],
                                        IrType::Ptr(Box::new(IrType::I8)), // Byte-based
                                    )?;
                                    let tag_ptr_i32 = self.builder.build_bitcast(
                                        tag_ptr,
                                        IrType::Ptr(Box::new(IrType::I32)),
                                    )?;
                                    let tag_val =
                                        self.builder.build_const(IrValue::I32(idx as i32))?;
                                    self.builder.build_store(tag_ptr_i32, tag_val)?;

                                    // Store each parameter at byte offset 8 + i*8
                                    // When is_method=true, args[0] is the enum class reference
                                    // (receiver), not a constructor field. Skip it.
                                    let constructor_args: &[HirExpr] = if *is_method {
                                        if args.len() > 1 {
                                            &args[1..]
                                        } else {
                                            &[]
                                        }
                                    } else {
                                        args
                                    };
                                    for (i, arg) in constructor_args.iter().enumerate() {
                                        let arg_reg = self.lower_expression(arg)?;
                                        let field_offset = self
                                            .builder
                                            .build_const(IrValue::I64((8 + i * 8) as i64))?;
                                        // Use I8 element type for byte-based addressing
                                        let field_ptr = self.builder.build_gep(
                                            ptr,
                                            vec![field_offset],
                                            IrType::Ptr(Box::new(IrType::I8)),
                                        )?;
                                        // Bitcast to i64 ptr for the store
                                        let field_ptr_i64 = self.builder.build_bitcast(
                                            field_ptr,
                                            IrType::Ptr(Box::new(IrType::I64)),
                                        )?;
                                        self.builder.build_store(field_ptr_i64, arg_reg)?;
                                    }

                                    // Return pointer as i64 for uniform handling
                                    // (bitcast pointer to i64)
                                    return self.builder.build_bitcast(ptr, IrType::I64);
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
