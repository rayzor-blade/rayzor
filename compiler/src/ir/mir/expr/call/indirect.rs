//! The terminal case: a call through a function pointer.

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
    pub(crate) fn lower_indirect_call(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::Call { callee, args, .. } = &expr.kind else {
            unreachable!("lower_indirect_call on a non-Call expression")
        };
        self.builder.call_label = Some("INDIRECT_CALL".to_string());

        debug!(
            "Taking indirect function call path - callee kind={:?}, args.len()={}",
            std::mem::discriminant(&callee.kind),
            args.len()
        );

        // Formal parameter types from the callee's function type. A `Void`
        // entry is Haxe's spelling for "takes nothing", not a slot.
        let formal_tys: Option<Vec<TypeId>> = {
            let type_table = self.type_table;
            type_table.get(callee.ty).and_then(|t| match &t.kind {
                crate::tast::TypeKind::Function { params, .. } => Some(
                    params
                        .iter()
                        .copied()
                        .filter(|p| {
                            !matches!(type_table.get(*p).map(|t| &t.kind), Some(TypeKind::Void))
                        })
                        .collect(),
                ),
                _ => None,
            })
        };

        // A function value has no function id for bind_skipped_optional_args.
        // Match a supplied argument to a later formal only when its shape
        // cannot fit the current optional slot.
        let arg_formals: Vec<usize> = {
            let type_table = self.type_table;
            let shape = |mut ty: TypeId| {
                for _ in 0..4 {
                    match type_table.get(ty).map(|t| &t.kind) {
                        Some(TypeKind::Optional { inner_type }) => ty = *inner_type,
                        Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                        _ => break,
                    }
                }
                match type_table.get(ty).map(|t| &t.kind) {
                    Some(TypeKind::Int | TypeKind::Float | TypeKind::Bool) => 1,
                    Some(TypeKind::String) => 2,
                    _ => 0,
                }
            };
            let mut next_formal = 0;
            args.iter()
                .map(|arg| {
                    if let Some(formals) = &formal_tys {
                        while next_formal + 1 < formals.len()
                            && matches!(
                                type_table.get(formals[next_formal]).map(|t| &t.kind),
                                Some(TypeKind::Optional { .. })
                            )
                            && shape(arg.ty) != 0
                            && shape(formals[next_formal]) != 0
                            && shape(arg.ty) != shape(formals[next_formal])
                            && shape(arg.ty) == shape(formals[next_formal + 1])
                        {
                            next_formal += 1;
                        }
                    }
                    let assigned = next_formal;
                    next_formal += 1;
                    assigned
                })
                .collect()
        };

        // Arguments are lowered before the callee, so lambdas passed as
        // arguments are still generated when callee lowering fails.
        debug!("About to lower {} indirect call arguments", args.len());
        // Every argument must lower. Dropping the ones that fail would keep the
        // call but shift the survivors into the wrong parameter slots, so a
        // miscompiled argument becomes a silently miscompiled call.
        let mut arg_regs: Vec<IrId> = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            debug!("  arg[{}] kind={:?}", i, std::mem::discriminant(&a.kind));
            let Some(reg) = self.lower_expression(a) else {
                warn!(
                    "indirect call: argument {} of {} failed to lower ({:?}); abandoning the call",
                    i,
                    args.len(),
                    std::mem::discriminant(&a.kind)
                );
                return None;
            };
            // A scalar or String handed to a `Dynamic` formal must box, as the
            // direct-call path does: the callee unboxes those. It does NOT unbox
            // a function, anonymous, class or enum value received as `Dynamic` —
            // it uses the raw pointer, so boxing those here breaks the callee.
            // An abstract travels as the type it stores.
            let storage_ty = self.resolve_storage_type(a.ty);
            let boxable = matches!(
                self.type_table.get(storage_ty).map(|t| &t.kind),
                Some(TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::String)
            );
            let reg = match formal_tys
                .as_ref()
                .and_then(|f| f.get(arg_formals[i]).copied())
            {
                Some(formal) if boxable => {
                    self.maybe_box_value(reg, storage_ty, formal).unwrap_or(reg)
                }
                Some(formal) => self
                    .unbox_optional_for_erased_formal(reg, a.ty, formal)
                    .unwrap_or(reg),
                None => reg,
            };
            arg_regs.push(reg);
        }
        debug!(
            "Lowered {} indirect call arguments successfully",
            arg_regs.len()
        );

        let func_ptr = self.lower_expression(callee)?;
        // A function held in a Dynamic is a box; the call goes to the
        // closure inside it, not to the box's tag.
        let func_ptr = self.unbox_dynamic_function(func_ptr, callee);

        // A callee typed Dynamic has no signature to call by: every argument
        // travels as a box to the closure's box-shaped entry, and the result
        // comes back as one.
        let callee_is_dynamic = matches!(
            self.type_table.get(callee.ty).map(|t| &t.kind),
            Some(TypeKind::Dynamic)
        );
        if callee_is_dynamic {
            return self.lower_dynamic_closure_call(func_ptr, args, &arg_regs);
        }

        // Generic slots carry bits; the registered entry adapts the concrete closure ABI.
        let uses_generic_slots = self.type_table.get(callee.ty).is_some_and(|ty| {
            let TypeKind::Function {
                params,
                return_type,
                ..
            } = &ty.kind
            else {
                return false;
            };
            params.iter().chain(std::iter::once(return_type)).any(|ty| {
                matches!(
                    self.type_table.get(*ty).map(|ty| &ty.kind),
                    Some(TypeKind::TypeParameter { .. })
                )
            })
        });
        if uses_generic_slots {
            return self.lower_generic_slot_call(
                expr,
                func_ptr,
                &arg_regs,
                formal_tys.as_deref(),
                &arg_formals,
            );
        }

        // Signature from the callee's function type, else from the arguments.
        let type_param_names = self.current_type_param_names();
        let param_types: Vec<IrType> = {
            let type_table = self.type_table;
            let callee_type = type_table.get(callee.ty);
            if let Some(type_ref) = callee_type {
                if let crate::tast::TypeKind::Function { params, .. } = &type_ref.kind {
                    // `Void -> T` is Haxe's spelling for "takes nothing", so a
                    // Void entry is notation, not a slot. Keeping it yields a
                    // parameter no call site can fill: LLVM rejects a Void
                    // parameter, and Cranelift asserts on the argument count.
                    params
                        .iter()
                        .map(|p| self.convert_type_or_type_var(*p, &type_param_names))
                        .filter(|t| !matches!(t, IrType::Void))
                        .collect()
                } else {
                    // Fallback: infer from actual argument types
                    args.iter().map(|a| self.convert_type(a.ty)).collect()
                }
            } else {
                args.iter().map(|a| self.convert_type(a.ty)).collect()
            }
        };
        let return_type = Box::new(self.convert_type(expr.ty));

        if arg_formals
            .iter()
            .enumerate()
            .any(|(arg, formal)| arg != *formal)
        {
            let supplied = std::mem::take(&mut arg_regs);
            for (formal, ty) in param_types.iter().enumerate() {
                if let Some((arg, _)) = arg_formals
                    .iter()
                    .enumerate()
                    .find(|(_, assigned)| **assigned == formal)
                {
                    arg_regs.push(supplied[arg]);
                } else if formal < arg_formals.last().copied().unwrap_or(0) {
                    arg_regs.push(self.zero_of(ty)?);
                } else {
                    break;
                }
            }
        }

        // Haxe accepts a short call only when the missing parameters are
        // optional; they travel as null (a zero of their slot type), and a
        // literal with a default applies it on receiving null.
        for missing in param_types.iter().skip(arg_regs.len()) {
            let value = self.zero_of(missing)?;
            arg_regs.push(value);
        }

        let func_signature = IrType::Function {
            params: param_types,
            return_type,
            varargs: false,
        };

        self.builder
            .build_call_indirect(func_ptr, arg_regs, func_signature)
    }

    fn lower_generic_slot_call(
        &mut self,
        expr: &HirExpr,
        closure: IrId,
        arg_regs: &[IrId],
        formal_tys: Option<&[TypeId]>,
        assigned: &[usize],
    ) -> Option<IrId> {
        let mut supplied = Vec::with_capacity(arg_regs.len());
        for reg in arg_regs {
            let ty = self.builder.get_register_type(*reg)?;
            let slot = match ty {
                IrType::I64 => *reg,
                IrType::F32 => {
                    let widened = self.builder.build_cast(*reg, IrType::F32, IrType::F64)?;
                    self.builder.build_bitcast(widened, IrType::I64)?
                }
                ty if ty.is_integer() => self.builder.build_cast(*reg, ty, IrType::I64)?,
                _ => self.builder.build_bitcast(*reg, IrType::I64)?,
            };
            supplied.push(slot);
        }
        let count = formal_tys
            .map_or(arg_regs.len(), |params| params.len())
            .max(assigned.last().map_or(0, |slot| slot + 1));
        let null = self.builder.build_const(IrValue::I64(0))?;
        let slots = (0..count)
            .map(|formal| {
                assigned
                    .iter()
                    .position(|slot| *slot == formal)
                    .map(|arg| supplied[arg])
                    .unwrap_or(null)
            })
            .collect::<Vec<_>>();
        let ptr = IrType::Ptr(Box::new(IrType::U8));
        let view = self.get_or_register_extern_function(
            "haxe_closure_slot_view",
            vec![ptr.clone()],
            ptr.clone(),
        );
        let closure = self.builder.build_call_direct(view, vec![closure], ptr)?;
        let expected = self.convert_type(expr.ty);
        let signature = IrType::Function {
            params: vec![IrType::I64; slots.len()],
            return_type: Box::new(if expected == IrType::Void {
                IrType::Void
            } else {
                IrType::I64
            }),
            varargs: false,
        };
        let result = self
            .builder
            .build_call_indirect(closure, slots, signature)?;
        match expected {
            IrType::I64 | IrType::Void => Some(result),
            IrType::F32 => {
                let value = self.builder.build_bitcast(result, IrType::F64)?;
                self.builder.build_cast(value, IrType::F64, IrType::F32)
            }
            ty if ty.is_integer() => self.builder.build_cast(result, IrType::I64, ty),
            ty => self.builder.build_bitcast(result, ty),
        }
    }

    /// `recv.m(args)` on a Dynamic or structurally typed receiver whose method
    /// the typer could not resolve: `m` is read by name at run time (a closure
    /// field, or a class method's bound thunk) and called through its
    /// box-shaped entry. On a Dynamic receiver, names of builtin container,
    /// string and iterator members keep their static binding, since the value
    /// may be an array or a string, which has no methods by name.
    pub(crate) fn try_dynamic_member_call(
        &mut self,
        expr: &HirExpr,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        const BUILTIN_MEMBERS: &[&str] = &[
            "push",
            "pop",
            "shift",
            "unshift",
            "insert",
            "remove",
            "indexOf",
            "lastIndexOf",
            "contains",
            "concat",
            "join",
            "reverse",
            "slice",
            "splice",
            "sort",
            "map",
            "filter",
            "iterator",
            "keyValueIterator",
            "copy",
            "resize",
            "toString",
            "charAt",
            "charCodeAt",
            "substr",
            "substring",
            "split",
            "toLowerCase",
            "toUpperCase",
            "get",
            "set",
            "exists",
            "keys",
            "clear",
            "hasNext",
            "next",
        ];
        const ITERATOR_MEMBERS: &[&str] = &["hasNext", "next", "iterator", "keyValueIterator"];
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            ..
        } = &expr.kind
        else {
            unreachable!("try_dynamic_member_call on a non-Call expression")
        };
        let method = match &callee.kind {
            HirExprKind::Variable { symbol, .. } if *is_method && !args.is_empty() => *symbol,
            _ => {
                *fell_through = true;
                return None;
            }
        };
        if self
            .dynamic_member_fallback
            .is_some_and(|(node, _)| node == &args[0] as *const HirExpr as usize)
        {
            *fell_through = true;
            return None;
        }
        // `super.m()` and `this.m()` are the class's own methods.
        let own_receiver = match &args[0].kind {
            HirExprKind::Super | HirExprKind::This => true,
            HirExprKind::Variable { symbol, .. } => self
                .symbol_table
                .get_symbol(*symbol)
                .is_some_and(|s| self.string_interner.get(s.name) == Some("this")),
            _ => false,
        };
        if own_receiver
            || self.function_map.contains_key(&method)
            || self.external_function_map.contains_key(&method)
        {
            *fell_through = true;
            return None;
        }
        let receiver_ty = self.resolve_through_aliases(args[0].ty);
        let (dynamic, structural) = match self.type_table.get(receiver_ty).map(|t| &t.kind) {
            Some(TypeKind::Dynamic) => (true, false),
            Some(TypeKind::Anonymous { .. }) => (false, true),
            _ => (false, false),
        };
        let name = self
            .symbol_table
            .get_symbol(method)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("");
        let excluded = (dynamic && BUILTIN_MEMBERS.contains(&name))
            || (structural && ITERATOR_MEMBERS.contains(&name));
        if !(dynamic || structural) || name.is_empty() || excluded {
            *fell_through = true;
            return None;
        }
        let dynamic_ty = self.type_table.dynamic_type();
        let receiver = self.lower_expression(&args[0])?;
        let member = if dynamic {
            self.dynamic_reflect_field_read(receiver, method, dynamic_ty)?
        } else {
            self.raw_anon_reflect_field_read(receiver, method, dynamic_ty)?
        };
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let member = match self.builder.get_register_type(member) {
            Some(IrType::Ptr(_)) | None => member,
            Some(other) => self.builder.build_cast(member, other, ptr_u8.clone())?,
        };
        // The runtime's TYPE_FUNCTION box holds the closure record.
        let unwrap = self.get_or_register_extern_function(
            "haxe_unbox_if_tag",
            vec![ptr_u8.clone(), IrType::U32],
            ptr_u8.clone(),
        );
        let function_tag = self.builder.build_const(IrValue::U32(u32::MAX - 1))?;
        let closure = self
            .builder
            .build_call_direct(unwrap, vec![member, function_tag], ptr_u8)?;
        // A receiver that is not an object with this member (a class value, a
        // null, a static extension's first argument) takes the statically
        // bound call, over the receiver already lowered.
        let null = self.builder.build_null()?;
        let found = self.builder.build_cmp(CompareOp::Ne, closure, null)?;
        let by_name = self.builder.create_block()?;
        let bound = self.builder.create_block()?;
        let merge = self.builder.create_block()?;
        self.builder.build_cond_branch(found, by_name, bound)?;
        let result_ty = self.convert_type(expr.ty);
        let is_void = matches!(result_ty, IrType::Void);

        self.builder.switch_to_block(by_name);
        let dynamic_result = self.call_member_closure(closure, &args[1..], expr)?;
        let dynamic_result = self.coerce_register(dynamic_result, &result_ty);
        let by_name_exit = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        self.builder.switch_to_block(bound);
        let outer = self
            .dynamic_member_fallback
            .replace((&args[0] as *const HirExpr as usize, receiver));
        let bound_result = self.lower_call(expr);
        self.dynamic_member_fallback = outer;
        // No value: a void call, or a member with no static binding.
        let bound_result = match bound_result {
            _ if is_void => None,
            Some(r) => Some(self.coerce_register(r, &result_ty)),
            None => Some(self.zero_of(&result_ty)?),
        };
        let bound_exit = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        self.builder.switch_to_block(merge);
        let Some(bound_result) = bound_result else {
            return None;
        };
        let phi = self.builder.build_phi(merge, result_ty)?;
        self.builder
            .add_phi_incoming(merge, phi, by_name_exit, dynamic_result);
        self.builder
            .add_phi_incoming(merge, phi, bound_exit, bound_result);
        if self.boxed_value_regs.contains(&dynamic_result) {
            self.boxed_value_regs.insert(phi);
        }
        Some(phi)
    }

    fn call_member_closure(
        &mut self,
        closure: IrId,
        call_args: &[HirExpr],
        expr: &HirExpr,
    ) -> Option<IrId> {
        let dynamic_ty = self.type_table.dynamic_type();
        let arg_regs: Vec<IrId> = call_args
            .iter()
            .map(|a| self.lower_expression(a))
            .collect::<Option<_>>()?;
        let result = self.lower_dynamic_closure_call(closure, call_args, &arg_regs)?;
        self.maybe_unbox_value(result, dynamic_ty, expr.ty)
    }

    fn zero_of(&mut self, ty: &IrType) -> Option<IrId> {
        match ty {
            IrType::F64 => self.builder.build_const(IrValue::F64(0.0)),
            IrType::F32 => self.builder.build_const(IrValue::F32(0.0)),
            IrType::Bool => self.builder.build_const(IrValue::Bool(false)),
            IrType::I32 => self.builder.build_const(IrValue::I32(0)),
            IrType::I64 => self.builder.build_const(IrValue::I64(0)),
            other => {
                let null = self.builder.build_const(IrValue::Null)?;
                self.builder.build_bitcast(null, other.clone())
            }
        }
    }

    fn coerce_register(&mut self, reg: IrId, ty: &IrType) -> IrId {
        match self.builder.get_register_type(reg) {
            Some(have) if &have != ty && !matches!(ty, IrType::Void) => self
                .builder
                .build_cast(reg, have, ty.clone())
                .unwrap_or(reg),
            _ => reg,
        }
    }

    /// `f(args)` with `f: Dynamic`: box the arguments, take the closure's
    /// box-shaped entry view and call it as `(box..) -> box`.
    fn lower_dynamic_closure_call(
        &mut self,
        closure: IrId,
        args: &[HirExpr],
        arg_regs: &[IrId],
    ) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let boxed_args = self.box_dynamic_call_args(args, arg_regs)?;
        let closure = match self.builder.get_register_type(closure) {
            Some(IrType::Ptr(_)) => closure,
            Some(other) => self
                .builder
                .build_cast(closure, other, ptr_u8.clone())
                .unwrap_or(closure),
            None => closure,
        };
        let is_varargs = self.get_or_register_extern_function(
            "haxe_closure_is_varargs",
            vec![ptr_u8.clone()],
            IrType::Bool,
        );
        let packed = self.builder.create_block()?;
        let direct = self.builder.create_block()?;
        let merge = self.builder.create_block()?;
        let flag = self
            .builder
            .build_call_direct(is_varargs, vec![closure], IrType::Bool)?;
        self.builder.build_cond_branch(flag, packed, direct)?;

        self.builder.switch_to_block(packed);
        let array = self.pack_dynamic_call_args(&boxed_args)?;
        let call = self.get_or_register_extern_function(
            "haxe_call_method_dynamic",
            vec![ptr_u8.clone(), ptr_u8.clone()],
            ptr_u8.clone(),
        );
        let packed_result =
            self.builder
                .build_call_direct(call, vec![closure, array], ptr_u8.clone())?;
        let packed_exit = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        self.builder.switch_to_block(direct);
        let view_fn = self.get_or_register_extern_function(
            "haxe_closure_dynamic_view",
            vec![ptr_u8.clone()],
            ptr_u8.clone(),
        );
        let view = self
            .builder
            .build_call_direct(view_fn, vec![closure], ptr_u8.clone())?;
        let signature = IrType::Function {
            params: vec![ptr_u8.clone(); boxed_args.len()],
            return_type: Box::new(ptr_u8.clone()),
            varargs: false,
        };
        let direct_result = self
            .builder
            .build_call_indirect(view, boxed_args, signature)?;
        let direct_exit = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        self.builder.switch_to_block(merge);
        let result = self.builder.build_phi(merge, ptr_u8)?;
        self.builder
            .add_phi_incoming(merge, result, packed_exit, packed_result)?;
        self.builder
            .add_phi_incoming(merge, result, direct_exit, direct_result)?;
        self.boxed_value_regs.insert(result);
        Some(result)
    }

    fn box_dynamic_call_args(&mut self, args: &[HirExpr], arg_regs: &[IrId]) -> Option<Vec<IrId>> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let dynamic_ty = self.type_table.dynamic_type();
        let mut boxed_args = Vec::with_capacity(arg_regs.len());
        for (a, reg) in args.iter().zip(arg_regs) {
            let boxed = if self.boxed_value_regs.contains(reg) {
                *reg
            } else {
                self.maybe_box_value(*reg, a.ty, dynamic_ty)?
            };
            let boxed = match self.builder.get_register_type(boxed) {
                Some(IrType::Ptr(_)) | None => boxed,
                Some(other) => self
                    .builder
                    .build_cast(boxed, other, ptr_u8.clone())
                    .unwrap_or(boxed),
            };
            boxed_args.push(boxed);
        }
        Some(boxed_args)
    }

    pub(crate) fn lower_dynamic_argument_array(
        &mut self,
        args: &[HirExpr],
        arg_regs: &[IrId],
    ) -> Option<IrId> {
        let boxed = self.box_dynamic_call_args(args, arg_regs)?;
        self.pack_dynamic_call_args(&boxed)
    }

    pub(crate) fn pack_dynamic_call_args(&mut self, args: &[IrId]) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let slots = self.builder.build_alloc(
            IrType::Array(Box::new(ptr_u8.clone()), args.len().max(1)),
            None,
        )?;
        for (i, value) in args.iter().enumerate() {
            let index = self.builder.build_const(IrValue::I64(i as i64))?;
            let slot = self.builder.build_gep(slots, vec![index], ptr_u8.clone())?;
            self.builder.build_store(slot, *value)?;
        }
        let count = self.builder.build_const(IrValue::U64(args.len() as u64))?;
        let array = self.get_or_register_extern_function(
            "haxe_array_from_dynamic_args",
            vec![ptr_u8.clone(), IrType::U64],
            ptr_u8.clone(),
        );
        self.builder
            .build_call_direct(array, vec![slots, count], ptr_u8)
    }
}
