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
    pub(crate) fn array_element_descriptor(&mut self, ty: TypeId) -> Option<(IrId, IrId)> {
        if let Some(name) = self.resolve_type_param_name(ty)
            && self.current_type_param_names().contains(&name)
        {
            return self.type_param_value_descriptor(name, 0);
        }
        let nullable_box = self.type_table.get(ty).is_some_and(|ty| match &ty.kind {
            TypeKind::Optional { inner_type } => {
                matches!(
                    self.type_table.get(*inner_type).map(|ty| &ty.kind),
                    Some(TypeKind::Float | TypeKind::Bool)
                )
            }
            _ => false,
        });
        let ty = if nullable_box {
            IrType::dynamic_ptr()
        } else {
            self.convert_type_arg(ty)
        };
        Some((
            self.builder
                .build_const(IrValue::I32(ty.value_tag() as i32))?,
            self.builder
                .build_const(IrValue::U64(ty.value_metadata()))?,
        ))
    }

    pub(crate) fn init_array_element_layout(&mut self, array: IrId, ty: TypeId) -> Option<()> {
        let element = match self.type_table.get(ty).map(|ty| &ty.kind) {
            Some(TypeKind::Array { element_type }) => *element_type,
            _ => return Some(()),
        };
        if !matches!(
            self.type_table.get(element).map(|ty| &ty.kind),
            Some(TypeKind::Dynamic | TypeKind::TypeParameter { .. })
        ) && !self.is_any_type(element)
        {
            return Some(());
        }
        let (tag, _) = self.array_element_descriptor(element)?;
        let function = self.get_or_register_extern_function(
            "haxe_array_init_element_tag",
            vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
            IrType::Void,
        );
        self.builder
            .build_call_direct(function, vec![array, tag], IrType::Void);
        Some(())
    }

    /// Keep the raw slot layout when a concrete array is exposed through an erased view.
    pub(crate) fn mark_int_array_for_erased_view(
        &mut self,
        reg: IrId,
        source: &HirExpr,
        target_ty: TypeId,
    ) {
        let element_of = |me: &Self, ty: TypeId| {
            me.type_table
                .get(me.resolve_storage_type(ty))
                .and_then(|t| match &t.kind {
                    TypeKind::Array { element_type } => Some(*element_type),
                    _ => None,
                })
        };
        let source_ty = match &source.kind {
            HirExprKind::Cast { expr, .. } if !matches!(expr.kind, HirExprKind::Array { .. }) => {
                expr.ty
            }
            _ => source.ty,
        };
        let Some(element) = element_of(self, source_ty) else {
            return;
        };
        let erased = element_of(self, target_ty).is_some_and(|element| {
            matches!(
                self.type_table.get(element).map(|t| &t.kind),
                Some(TypeKind::Dynamic)
            ) || self.is_any_type(element)
        });
        if !erased {
            return;
        }
        let layout = self.erased_array_slot_layout(element, 0);
        if layout == 0 {
            return;
        }
        let ptr = IrType::Ptr(Box::new(IrType::U8));
        let Some(array) = self.coerce_reg_to(reg, &ptr) else {
            return;
        };
        let Some(layout) = self.builder.build_const(IrValue::U64(layout)) else {
            return;
        };
        let function = self.get_or_register_extern_function(
            "haxe_array_mark_slot_layout",
            vec![ptr, IrType::U64],
            IrType::Void,
        );
        self.builder
            .build_call_direct(function, vec![array, layout], IrType::Void);
    }

    fn lower_array_storage_call(
        &mut self,
        expr: &HirExpr,
        name: &str,
        args: &[HirExpr],
    ) -> Option<IrId> {
        let element = self
            .type_table
            .get(self.resolve_storage_type(args[0].ty))
            .and_then(|ty| match &ty.kind {
                TypeKind::Array { element_type } => Some(*element_type),
                _ => None,
            })?;
        let ptr = IrType::Ptr(Box::new(IrType::U8));
        let array = self.lower_expression(&args[0])?;
        let array = self.coerce_reg_to(array, &ptr)?;
        if matches!(name, "pop" | "shift") {
            let target = if self.is_null_int_type(expr.ty) {
                expr.ty
            } else {
                element
            };
            let (tag, _) = self.array_element_descriptor(target)?;
            let from_end = self.builder.build_const(IrValue::Bool(name == "pop"))?;
            let offset = self.builder.build_const(IrValue::I64(8))?;
            let length_ptr = self.builder.build_ptr_add(array, offset, ptr.clone())?;
            let length = self.builder.build_load(length_ptr, IrType::I64)?;
            let zero = self.builder.build_const(IrValue::I64(0))?;
            let present = self.builder.build_cmp(CompareOp::Ne, length, zero)?;
            let function = self.get_or_register_extern_function(
                "haxe_array_take_typed",
                vec![ptr, IrType::Bool, IrType::I32],
                IrType::I64,
            );
            let value = self.builder.build_call_direct(
                function,
                vec![array, from_end, tag],
                IrType::I64,
            )?;
            let result_type = self.convert_type(expr.ty);
            let result_type = if result_type == IrType::Void {
                self.convert_type(element)
            } else {
                result_type
            };
            let value = if self.is_null_int_type(expr.ty) {
                self.coerce_reg_to(value, &result_type)?
            } else {
                let element_type = self.convert_type(element);
                let value = self.coerce_reg_to(value, &element_type)?;
                let value = self
                    .maybe_box_value(value, element, expr.ty)
                    .unwrap_or(value);
                self.coerce_reg_to(value, &result_type)?
            };
            if matches!(result_type, IrType::Ptr(_)) {
                let bits = self.coerce_reg_to(value, &IrType::I64)?;
                let bits = self.builder.build_select(present, bits, zero)?;
                self.coerce_reg_to(bits, &result_type)
            } else {
                Some(value)
            }
        } else {
            let arg = args.last()?;
            let value = self.lower_expression(arg)?;
            let value = self
                .maybe_unbox_value(value, arg.ty, element)
                .unwrap_or(value);
            let value = self
                .maybe_box_value(value, arg.ty, element)
                .unwrap_or(value);
            let ty = self.convert_type(element);
            let value = self.coerce_reg_to(value, &ty)?;
            if name == "push" && ty == IrType::F64 {
                let function = self.get_or_register_extern_function(
                    "haxe_array_push_typed_f64",
                    vec![ptr, IrType::F64],
                    IrType::I32,
                );
                return self
                    .builder
                    .build_call_direct(function, vec![array, value], IrType::I32);
            }
            let (tag, metadata) = self.array_element_descriptor(element)?;
            let bits = self.coerce_reg_to(value, &IrType::I64)?;
            let mut types = vec![ptr];
            let mut values = vec![array];
            if name == "insert" {
                let index = self.lower_expression(&args[1])?;
                let index = self.coerce_reg_to(index, &IrType::I32)?;
                types.push(IrType::I32);
                values.push(index);
            }
            types.extend([IrType::I64, IrType::I32, IrType::U64]);
            values.extend([bits, tag, metadata]);
            let runtime = match name {
                "push" => "haxe_array_push_typed",
                "unshift" => "haxe_array_unshift_typed",
                _ => "haxe_array_insert_typed",
            };
            let result = if name == "push" {
                IrType::I32
            } else {
                IrType::Void
            };
            let function = self.get_or_register_extern_function(runtime, types, result.clone());
            self.builder.build_call_direct(function, values, result)
        }
    }

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
        if *is_method
            && matches!(
                (vname, args.len()),
                ("push" | "unshift", 2) | ("insert", 3) | ("pop" | "shift", 1)
            )
            && args.first().is_some_and(|arg| {
                matches!(
                    self.type_table
                        .get(self.resolve_storage_type(arg.ty))
                        .map(|ty| &ty.kind),
                    Some(TypeKind::Array { .. })
                )
            })
        {
            return self.lower_array_storage_call(expr, vname, args);
        }
        if *is_method
            && Self::null_int_array_method(vname, args.len())
            && self.array_of_null_int(args[0].ty)
        {
            return self.lower_null_int_array_call(vname, args);
        }
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
            let dynamic_search = element_type.is_some_and(|ty| is_dynamic(self, ty));
            // An erased `Array<T>` searched for a `T` compares the way `==` on
            // T does, by the tag monomorphisation substitutes for T.
            let type_param_name = |me: &Self, ty: TypeId| match me.type_table.get(ty) {
                Some(t) => match &t.kind {
                    TypeKind::TypeParameter { symbol_id, .. } => me
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|s| me.string_interner.get(s.name))
                        .map(str::to_string),
                    _ => None,
                },
                None => None,
            };
            let shared_param = element_type
                .and_then(|ty| type_param_name(self, ty))
                .filter(|name| type_param_name(self, args[1].ty).as_ref() == Some(name));
            let scalar_search = element_type.is_some_and(|ty| {
                matches!(
                    self.convert_type(ty),
                    IrType::I32 | IrType::I64 | IrType::F64 | IrType::Bool | IrType::NullInt
                )
            });
            let typed_search = shared_param.is_some() || scalar_search;
            let search = element_type.and_then(|ty| {
                if typed_search {
                    Some(("haxe_array_typed_index_of", IrType::I64))
                } else if self.convert_type(ty) == IrType::String {
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
                let value = if typed_search {
                    let element = element_type?;
                    let value = self
                        .maybe_unbox_value(value, args[1].ty, element)
                        .unwrap_or(value);
                    let expected = self.convert_type(element);
                    let value = self.coerce_reg_to(value, &expected)?;
                    self.erase_reflect_compare_arg(value)
                } else {
                    let value = if dynamic_search {
                        self.maybe_box_value(value, args[1].ty, element_type?)?
                    } else {
                        value
                    };
                    self.coerce_reg_to(value, &value_type)?
                };
                let from = if let Some(arg) = args.get(2) {
                    self.lower_expression(arg)?
                } else {
                    self.builder
                        .build_const(IrValue::I64(if vname == "lastIndexOf" { -1 } else { 0 }))?
                };
                let reverse = self
                    .builder
                    .build_const(IrValue::I32(i32::from(vname == "lastIndexOf")))?;
                let mut param_types = vec![
                    IrType::Ptr(Box::new(IrType::U8)),
                    value_type,
                    IrType::I64,
                    IrType::I32,
                ];
                let mut call_args = vec![arr, value, from, reverse];
                if typed_search {
                    let (tag, _) = self.array_element_descriptor(element_type?)?;
                    param_types.push(IrType::I32);
                    call_args.push(tag);
                }
                let function = self.get_or_register_extern_function(name, param_types, IrType::I64);
                let index = self
                    .builder
                    .build_call_direct(function, call_args, IrType::I64)?;
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
            let null_int_elements = self.array_of_null_int(args[0].ty);
            let elem_tag: i32 = if null_int_elements {
                rayzor_runtime::type_system::ValueTag::NullInt as i32
            } else {
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

    /// The `Array<Null<Int>>` methods `lower_null_int_array_call` lowers, by
    /// name and argument count (the receiver included).
    fn null_int_array_method(name: &str, argc: usize) -> bool {
        matches!(
            (name, argc),
            ("push" | "unshift" | "contains" | "remove" | "resize", 2)
                | ("indexOf" | "lastIndexOf", 2 | 3)
                | ("insert", 3)
                | ("pop" | "shift" | "toString", 1)
        )
    }

    /// An `Array<Null<Int>>` method: an element crosses as its slot bits, so
    /// a search compares null with null and nothing else; `pop`/`shift` on
    /// an empty array and slots opened by `resize` read as null, and
    /// `toString` prints them as null.
    fn lower_null_int_array_call(&mut self, name: &str, args: &[HirExpr]) -> Option<IrId> {
        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
        let arr = self.lower_expression(&args[0])?;
        let element = |this: &mut Self, arg: &HirExpr| -> Option<IrId> {
            let reg = this.lower_expression(arg)?;
            this.null_int_slot_bits(reg, arg.ty)
        };
        let int_arg = |this: &mut Self, arg: &HirExpr, want: IrType| -> Option<IrId> {
            let reg = this.lower_expression(arg)?;
            if this.builder.get_register_type(reg) == Some(IrType::NullInt) {
                return this.null_int_to_ir(reg, &want);
            }
            this.coerce_reg_to(reg, &want)
        };
        match name {
            "push" => {
                let bits = element(self, &args[1])?;
                let push = self.get_or_register_extern_function(
                    "haxe_array_push_i64",
                    vec![IrType::Ptr(Box::new(IrType::I64)), IrType::I64],
                    IrType::Void,
                );
                self.builder
                    .build_call_direct(push, vec![arr, bits], IrType::Void);
                // The new length, read from the header.
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let len_off = self.builder.build_const(IrValue::I64(8))?;
                let len_ptr = self.builder.build_ptr_add(arr, len_off, ptr_u8)?;
                let len = self.builder.build_load(len_ptr, IrType::I64)?;
                self.builder.build_cast(len, IrType::I64, IrType::I32)
            }
            "pop" | "shift" => {
                let runtime = if name == "pop" {
                    "haxe_array_pop_null_int"
                } else {
                    "haxe_array_shift_null_int"
                };
                let function =
                    self.get_or_register_extern_function(runtime, vec![ptr_void], IrType::NullInt);
                self.builder
                    .build_call_direct(function, vec![arr], IrType::NullInt)
            }
            "toString" => {
                let string_ptr = IrType::Ptr(Box::new(IrType::String));
                let function = self.get_or_register_extern_function(
                    "haxe_array_to_string_i64",
                    vec![ptr_void],
                    string_ptr.clone(),
                );
                self.builder
                    .build_call_direct(function, vec![arr], string_ptr)
            }
            "unshift" => {
                let bits = element(self, &args[1])?;
                let unshift = self.get_or_register_extern_function(
                    "haxe_array_unshift",
                    vec![ptr_void, IrType::I64],
                    IrType::Void,
                );
                self.builder
                    .build_call_direct(unshift, vec![arr, bits], IrType::Void)
            }
            "insert" => {
                let pos = int_arg(self, &args[1], IrType::I32)?;
                let bits = element(self, &args[2])?;
                let insert = self.register_stdlib_mir_forward_ref(
                    "array_insert",
                    vec![ptr_void, IrType::I32, IrType::I64],
                    IrType::Void,
                );
                self.builder
                    .build_call_direct(insert, vec![arr, pos, bits], IrType::Void)
            }
            "resize" => {
                let len = int_arg(self, &args[1], IrType::I64)?;
                let resize = self.get_or_register_extern_function(
                    "haxe_array_resize_null_int",
                    vec![ptr_void, IrType::I64],
                    IrType::Void,
                );
                self.builder
                    .build_call_direct(resize, vec![arr, len], IrType::Void)
            }
            "remove" => {
                let bits = element(self, &args[1])?;
                let remove = self.get_or_register_extern_function(
                    "haxe_array_remove_value",
                    vec![ptr_void, IrType::I64],
                    IrType::Bool,
                );
                self.builder
                    .build_call_direct(remove, vec![arr, bits], IrType::Bool)
            }
            _ => {
                // indexOf, lastIndexOf, contains.
                let bits = element(self, &args[1])?;
                let (runtime, default_from) = if name == "lastIndexOf" {
                    ("haxe_array_last_index_of", i64::MAX)
                } else {
                    ("haxe_array_index_of", 0)
                };
                let from = match args.get(2) {
                    Some(arg) => int_arg(self, arg, IrType::I64)?,
                    None => self.builder.build_const(IrValue::I64(default_from))?,
                };
                let search = self.get_or_register_extern_function(
                    runtime,
                    vec![ptr_void, IrType::I64, IrType::I64],
                    IrType::I64,
                );
                let index =
                    self.builder
                        .build_call_direct(search, vec![arr, bits, from], IrType::I64)?;
                if name == "contains" {
                    let zero = self.builder.build_const(IrValue::I64(0))?;
                    return self.builder.build_cmp(CompareOp::Ge, index, zero);
                }
                self.builder.build_cast(index, IrType::I64, IrType::I32)
            }
        }
    }
}
