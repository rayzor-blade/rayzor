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
            let boxable = {
                let type_table = self.type_table;
                matches!(
                    type_table.get(a.ty).map(|t| &t.kind),
                    Some(TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::String)
                )
            };
            let reg = match formal_tys.as_ref().and_then(|f| f.get(i).copied()) {
                Some(formal) if boxable => self.maybe_box_value(reg, a.ty, formal).unwrap_or(reg),
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

        // Signature from the callee's function type, else from the arguments.
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
                        .map(|p| self.convert_type(*p))
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
        let dynamic_ty = self.type_table.dynamic_type();
        let mut boxed_args = Vec::with_capacity(arg_regs.len());
        for (a, reg) in args.iter().zip(arg_regs) {
            let boxed = self.maybe_box_value(*reg, a.ty, dynamic_ty).unwrap_or(*reg);
            let boxed = match self.builder.get_register_type(boxed) {
                Some(IrType::Ptr(_)) | None => boxed,
                Some(other) => self
                    .builder
                    .build_cast(boxed, other, ptr_u8.clone())
                    .unwrap_or(boxed),
            };
            boxed_args.push(boxed);
        }
        let closure = match self.builder.get_register_type(closure) {
            Some(IrType::Ptr(_)) => closure,
            Some(other) => self
                .builder
                .build_cast(closure, other, ptr_u8.clone())
                .unwrap_or(closure),
            None => closure,
        };
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
            return_type: Box::new(ptr_u8),
            varargs: false,
        };
        let result = self
            .builder
            .build_call_indirect(view, boxed_args, signature)?;
        self.boxed_value_regs.insert(result);
        Some(result)
    }
}
