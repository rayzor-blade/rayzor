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
        if matches!(vname, "contains" | "indexOf" | "lastIndexOf" | "remove")
            && *is_method
            && args.len() >= 2
        {
            let element_type = self
                .type_table
                .get(self.resolve_through_aliases(args[0].ty))
                .and_then(|t| match &t.kind {
                    TypeKind::Array { element_type } => Some(*element_type),
                    _ => None,
                });
            // A Dynamic searched value is a box, so a function box can be
            // matched by closure equality.
            let is_dynamic = |me: &Self, ty: TypeId| {
                matches!(
                    me.type_table
                        .get(me.resolve_through_aliases(ty))
                        .map(|t| &t.kind),
                    Some(TypeKind::Dynamic)
                )
            };
            let dynamic_search =
                element_type.is_some_and(|ty| is_dynamic(self, ty)) && is_dynamic(self, args[1].ty);
            let search = element_type.and_then(|ty| {
                if self.convert_type(ty) == IrType::String {
                    Some(("haxe_array_string_index_of", IrType::String))
                } else if matches!(
                    self.type_table
                        .get(self.resolve_through_aliases(ty))
                        .map(|t| &t.kind),
                    Some(TypeKind::Function { .. })
                ) {
                    Some((
                        "haxe_array_function_index_of",
                        IrType::Ptr(Box::new(IrType::U8)),
                    ))
                } else if dynamic_search {
                    Some((
                        "haxe_array_dynamic_index_of",
                        IrType::Ptr(Box::new(IrType::U8)),
                    ))
                } else {
                    None
                }
            });
            if let Some((name, value_type)) = search {
                let arr = self.lower_expression(&args[0])?;
                let value = self.lower_expression(&args[1])?;
                let value = self.coerce_reg_to(value, &value_type)?;
                let from = if let Some(arg) = args.get(2) {
                    self.lower_expression(arg)?
                } else {
                    self.builder
                        .build_const(IrValue::I64(if vname == "lastIndexOf" { -1 } else { 0 }))?
                };
                let reverse = self
                    .builder
                    .build_const(IrValue::I32(i32::from(vname == "lastIndexOf")))?;
                let function = self.get_or_register_extern_function(
                    name,
                    vec![
                        IrType::Ptr(Box::new(IrType::U8)),
                        value_type,
                        IrType::I64,
                        IrType::I32,
                    ],
                    IrType::I64,
                );
                let index = self.builder.build_call_direct(
                    function,
                    vec![arr, value, from, reverse],
                    IrType::I64,
                )?;
                return match vname {
                    "contains" => {
                        let zero = self.builder.build_const(IrValue::I64(0))?;
                        self.builder.build_cmp(CompareOp::Ge, index, zero)
                    }
                    // Remove the first matching element; false when none.
                    "remove" => {
                        let zero = self.builder.build_const(IrValue::I64(0))?;
                        let found = self.builder.build_cmp(CompareOp::Ge, index, zero)?;
                        let clamped = self.builder.build_select(found, index, zero)?;
                        let remove = self.get_or_register_extern_function(
                            "haxe_array_remove_at",
                            vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I64, IrType::Bool],
                            IrType::Bool,
                        );
                        self.builder.build_call_direct(
                            remove,
                            vec![arr, clamped, found],
                            IrType::Bool,
                        )
                    }
                    _ => Some(index),
                };
            }
        }

        // Array<Float>.push on WASM32 (Variable-callee shape, where the
        // receiver is desugared to args[0] and the value to args[1]).
        // The generic `array_push` MIR wrapper takes an I64 value param,
        // but IrType::I64 lowers to a WASM `i32`, so a Float value (the
        // f64 bit-pattern bitcast to i64) loses its high 32 bits and reads
        // back as 0/garbage. Route Float-element pushes through the f64
        // runtime entry — value stays F64 (→ WASM f64, full 8 bytes). This
        // matches the array-literal lowering and is bit-identical on native.
        if vname == "push" && *is_method && args.len() == 2 {
            let (elem_is_f64, elem_is_dynamic) = {
                let type_table = self.type_table;
                type_table
                    .get(self.resolve_through_aliases(args[0].ty))
                    .and_then(|t| {
                        if let TypeKind::Array { element_type } = &t.kind {
                            Some(*element_type)
                        } else {
                            None
                        }
                    })
                    .map(|et| {
                        (
                            self.convert_type(et) == IrType::F64,
                            matches!(type_table.get(et).map(|t| &t.kind), Some(TypeKind::Dynamic)),
                        )
                    })
                    .unwrap_or((false, false))
            };
            // Dynamic string slots need a tag for value-based equality.
            if elem_is_dynamic && self.convert_type(args[1].ty) == IrType::String {
                let arr = self.lower_expression(&args[0])?;
                let value = self.lower_expression(&args[1])?;
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let raw = self.builder.build_bitcast(value, ptr_u8.clone())?;
                let box_fn = self.get_or_register_extern_function(
                    "haxe_box_haxestring_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8,
                );
                let boxed = self.builder.build_call_direct(
                    box_fn,
                    vec![raw],
                    IrType::Ptr(Box::new(IrType::U8)),
                )?;
                let slot = self.builder.build_bitcast(boxed, IrType::I64)?;
                let push = self.get_or_register_extern_function(
                    "haxe_array_push_i64",
                    vec![IrType::Ptr(Box::new(IrType::I64)), IrType::I64],
                    IrType::Void,
                );
                return self
                    .builder
                    .build_call_direct(push, vec![arr, slot], IrType::Void);
            }
            // A Dynamic pushed into an Int or Bool array stores the scalar its
            // box holds; raw bits pass through unchanged.
            let elem_is_int_or_bool = self
                .type_table
                .get(self.resolve_through_aliases(args[0].ty))
                .and_then(|t| match &t.kind {
                    TypeKind::Array { element_type } => Some(*element_type),
                    _ => None,
                })
                .is_some_and(|et| {
                    matches!(
                        self.type_table.get(et).map(|t| &t.kind),
                        Some(TypeKind::Int | TypeKind::Bool)
                    )
                });
            let arg_is_dynamic = matches!(
                self.type_table
                    .get(self.resolve_through_aliases(args[1].ty))
                    .map(|t| &t.kind),
                Some(TypeKind::Dynamic)
            );
            if elem_is_int_or_bool && arg_is_dynamic {
                let arr = self.lower_expression(&args[0])?;
                let value = self.lower_expression(&args[1])?;
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let value = self.coerce_reg_to(value, &ptr_u8)?;
                let unbox = self.get_or_register_extern_function(
                    "haxe_unbox_scalar_or_addr",
                    vec![ptr_u8],
                    IrType::I64,
                );
                let slot = self
                    .builder
                    .build_call_direct(unbox, vec![value], IrType::I64)?;
                let push = self.get_or_register_extern_function(
                    "haxe_array_push_i64",
                    vec![IrType::Ptr(Box::new(IrType::I64)), IrType::I64],
                    IrType::Void,
                );
                return self
                    .builder
                    .build_call_direct(push, vec![arr, slot], IrType::Void);
            }
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
        // Only an array's (or an unknown receiver's) `join`: a user class
        // or abstract may declare its own. A structure's static fallback
        // runs once the receiver proved not to be an anon object with the
        // member, so there it is an array.
        let structural_fallback = args.first().is_some_and(|a| {
            self.dynamic_member_fallback
                .is_some_and(|(node, _)| node == a as *const HirExpr as usize)
                && matches!(
                    self.type_table
                        .get(self.resolve_through_aliases(a.ty))
                        .map(|t| &t.kind),
                    Some(TypeKind::Anonymous { .. })
                )
        });
        let receiver_is_arrayish = structural_fallback
            || args
                .first()
                .and_then(|a| self.type_table.get(a.ty))
                .is_none_or(|t| {
                    !matches!(
                        t.kind,
                        TypeKind::Class { .. }
                            | TypeKind::Abstract { .. }
                            | TypeKind::Interface { .. }
                            | TypeKind::Anonymous { .. }
                            | TypeKind::TypeAlias { .. }
                    )
                });
        if vname == "join" && *is_method && args.len() == 2 && receiver_is_arrayish {
            let enum_element = self
                .type_table
                .get(args[0].ty)
                .and_then(|t| match &t.kind {
                    TypeKind::Array { element_type } => Some(*element_type),
                    _ => None,
                })
                .and_then(|element| self.type_table.get(self.resolve_through_aliases(element)))
                .and_then(|t| match &t.kind {
                    TypeKind::Enum { symbol_id, .. } => Some(*symbol_id),
                    _ => None,
                })
                .or_else(|| {
                    let HirExprKind::Call {
                        target: CallTarget::Static { class, method },
                        type_args,
                        args: enum_args,
                        ..
                    } = &args[0].kind
                    else {
                        return None;
                    };
                    let names = [*class, *method].map(|symbol| {
                        self.symbol_table
                            .get_symbol(symbol)
                            .and_then(|s| self.string_interner.get(s.name))
                    });
                    if names != [Some("Type"), Some("allEnums")] {
                        return None;
                    }
                    type_args
                        .iter()
                        .chain(enum_args.first().map(|arg| &arg.ty))
                        .find_map(|ty| {
                            self.type_table
                                .get(self.resolve_through_aliases(*ty))
                                .and_then(|t| match &t.kind {
                                    TypeKind::Enum { symbol_id, .. } => Some(*symbol_id),
                                    _ => None,
                                })
                        })
                        .or_else(|| {
                            enum_args.first().and_then(|arg| match &arg.kind {
                                HirExprKind::Variable { symbol, .. }
                                    if self.symbol_table.get_symbol(*symbol).is_some_and(|s| {
                                        s.kind == crate::tast::symbols::SymbolKind::Enum
                                    }) =>
                                {
                                    Some(*symbol)
                                }
                                _ => None,
                            })
                        })
                });
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
                    .and_then(|et| {
                        type_table
                            .get(self.resolve_through_aliases(et))
                            .map(|t| t.kind.clone())
                    })
                    .map(|k| match k {
                        TypeKind::Int => 1,
                        TypeKind::Bool => 2,
                        TypeKind::Float => 4,
                        TypeKind::String => 5,
                        _ => 6,
                    })
                    // A receiver whose elements are not statically known (a
                    // Dynamic) takes the tag that lets each element say what
                    // it is; the String tag dereferenced a raw Int.
                    .unwrap_or(6)
            };
            if let (Some(arr_reg), Some(sep_reg)) = (
                self.lower_expression(&args[0]),
                self.lower_expression(&args[1]),
            ) {
                // A Dynamic receiver arrives boxed; the runtime wants the array.
                let arr_reg = self.unbox_dynamic_receiver(arr_reg, &args[0], "Array");
                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                let joined = if let Some(enum_symbol) = enum_element {
                    let type_id = self
                        .builder
                        .build_const(IrValue::U32(self.enum_runtime_id(enum_symbol)))?;
                    let is_boxed = self
                        .builder
                        .build_const(IrValue::I32(i32::from(self.enum_is_boxed(enum_symbol))))?;
                    let join_fn = self.get_or_register_extern_function(
                        "haxe_array_join_enum",
                        vec![ptr_void.clone(), ptr_void.clone(), IrType::U32, IrType::I32],
                        ptr_void.clone(),
                    );
                    self.builder.build_call_direct(
                        join_fn,
                        vec![arr_reg, sep_reg, type_id, is_boxed],
                        ptr_void.clone(),
                    )?
                } else {
                    let tag_reg = self.builder.build_const(IrValue::I32(elem_tag))?;
                    let join_fn = self.get_or_register_extern_function(
                        "haxe_array_join_typed",
                        vec![ptr_void.clone(), ptr_void.clone(), IrType::I32],
                        ptr_void.clone(),
                    );
                    self.builder.build_call_direct(
                        join_fn,
                        vec![arr_reg, sep_reg, tag_reg],
                        ptr_void.clone(),
                    )?
                };
                // Through a Dynamic receiver the result is Dynamic too, and a
                // Dynamic is a box.
                let receiver_is_dynamic = matches!(
                    self.type_table.get(args[0].ty).map(|t| &t.kind),
                    Some(TypeKind::Dynamic)
                );
                if receiver_is_dynamic {
                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                    let as_ptr = self.builder.build_bitcast(joined, ptr_u8.clone())?;
                    let box_fn = self.get_or_register_extern_function(
                        "haxe_box_haxestring_ptr",
                        vec![ptr_u8.clone()],
                        ptr_u8.clone(),
                    );
                    return self.builder.build_call_direct(box_fn, vec![as_ptr], ptr_u8);
                }
                return Some(joined);
            }
        }
        *fell_through = true;
        None
    }
}
