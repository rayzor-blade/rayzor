//! Method calls on a Dynamic receiver that more than one runtime class could
//! answer, resolved by the box's tag at runtime.

use super::*;
use crate::ir::{IrBlockId, IrId, IrType, IrValue};
use crate::stdlib::{MethodSignature, RuntimeFunctionCall};

impl<'a> HirToMirContext<'a> {
    /// The tag the box of `class` carries: the runtime's TYPE_ARRAY for
    /// `Array`, else the same hash the boxing side derives from the
    /// qualified name.
    fn stdlib_class_tag(&self, class: &str) -> u32 {
        if class == "Array" {
            7
        } else {
            Self::fnv1a_class_type_id(class)
        }
    }

    /// The receiver of a stdlib call resolved by name on a Dynamic: a box by
    /// convention, so the class's own function gets the value inside. An
    /// array or string is told from its box by the tag; a class box is
    /// unwrapped as one. A register with a class hint is a raw object from a
    /// wrapper's return and is left alone, as is any non-Dynamic receiver.
    pub(crate) fn unbox_dynamic_receiver(
        &mut self,
        reg: IrId,
        receiver: &HirExpr,
        class: &str,
    ) -> IrId {
        // Bound by a Let that boxed it for a Dynamic annotation the typer
        // then narrowed (`var m:Dynamic = new Map()`): a box whatever class
        // hint the register carries.
        let let_boxed = matches!(&receiver.kind, HirExprKind::Variable { symbol, .. }
            if self.boxed_dynamic_symbols.contains(symbol));
        let is_dynamic = matches!(
            self.type_table.get(receiver.ty).map(|t| &t.kind),
            Some(TypeKind::Dynamic)
        );
        if !let_boxed && (!is_dynamic || self.register_class_hints.contains_key(&reg)) {
            return reg;
        }
        if !matches!(self.builder.get_register_type(reg), Some(IrType::Ptr(_))) {
            return reg;
        }
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let tag = match class {
            "Array" => Some(7u32),
            "String" => Some(5u32),
            _ => None,
        };
        let out = match tag {
            Some(tag) => {
                let unbox = self.get_or_register_extern_function(
                    "haxe_unbox_if_tag",
                    vec![ptr_u8.clone(), IrType::U32],
                    ptr_u8.clone(),
                );
                let Some(tag_reg) = self.builder.build_const(IrValue::U32(tag)) else {
                    return reg;
                };
                self.builder
                    .build_call_direct(unbox, vec![reg, tag_reg], ptr_u8)
            }
            None => {
                let unbox = self.get_or_register_extern_function(
                    "haxe_unbox_reference_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8.clone(),
                );
                self.builder.build_call_direct(unbox, vec![reg], ptr_u8)
            }
        };
        out.unwrap_or(reg)
    }

    /// The closure inside a function box held in a Dynamic, for an indirect
    /// call; any other register is returned as it is.
    pub(crate) fn unbox_dynamic_function(&mut self, reg: IrId, callee: &HirExpr) -> IrId {
        let let_boxed = matches!(&callee.kind, HirExprKind::Variable { symbol, .. }
            if self.boxed_dynamic_symbols.contains(symbol));
        let is_dynamic = matches!(
            self.type_table.get(callee.ty).map(|t| &t.kind),
            Some(TypeKind::Dynamic)
        );
        if !let_boxed && !is_dynamic {
            return reg;
        }
        if !matches!(self.builder.get_register_type(reg), Some(IrType::Ptr(_))) {
            return reg;
        }
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let unbox = self.get_or_register_extern_function(
            "haxe_unbox_if_tag",
            vec![ptr_u8.clone(), IrType::U32],
            ptr_u8.clone(),
        );
        // The runtime's TYPE_FUNCTION.
        let Some(tag) = self.builder.build_const(IrValue::U32(u32::MAX - 1)) else {
            return reg;
        };
        self.builder
            .build_call_direct(unbox, vec![reg, tag], ptr_u8)
            .unwrap_or(reg)
    }

    /// A call on a Dynamic receiver among `candidates`, one per runtime class
    /// that declares the method. The receiver's box carries the class it was
    /// boxed with, so the call is bound at runtime: each candidate compares
    /// its tag and calls its own function on the unboxed value; no match
    /// throws. Every result comes back as a Dynamic box so the value has one
    /// shape whichever branch produced it.
    ///
    /// The receiver and arguments are lowered once, before the branches.
    pub(crate) fn dispatch_dynamic_call_by_tag(
        &mut self,
        args: &[HirExpr],
        method_name: &str,
        candidates: &[(&'static str, &MethodSignature, &RuntimeFunctionCall)],
        no_match_message: &str,
    ) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let recv = self.lower_expression(&args[0])?;
        let mut rest = Vec::with_capacity(args.len().saturating_sub(1));
        for arg in &args[1..] {
            rest.push((self.lower_expression(arg)?, arg.ty));
        }

        let tag_fn = self.get_or_register_extern_function(
            "haxe_dynamic_tag",
            vec![ptr_u8.clone()],
            IrType::U32,
        );
        let tag = self
            .builder
            .build_call_direct(tag_fn, vec![recv], IrType::U32)?;
        let unbox_fn = self.get_or_register_extern_function(
            "haxe_unbox_reference_ptr",
            vec![ptr_u8.clone()],
            ptr_u8.clone(),
        );

        let merge = self.builder.create_block()?;
        let result = self.builder.build_phi(merge, ptr_u8.clone())?;

        let mut seen: Vec<&str> = Vec::new();
        for &(class, _sig, call) in candidates {
            if seen.contains(&call.runtime_name) {
                continue;
            }
            seen.push(call.runtime_name);
            let hit = self.builder.create_block()?;
            let miss = self.builder.create_block()?;
            let expected = self
                .builder
                .build_const(IrValue::U32(self.stdlib_class_tag(class)))?;
            let is = self
                .builder
                .build_cmp(crate::ir::CompareOp::Eq, tag, expected)?;
            self.builder.build_cond_branch(is, hit, miss)?;

            self.builder.switch_to_block(hit);
            let unboxed = self
                .builder
                .build_call_direct(unbox_fn, vec![recv], ptr_u8.clone())?;
            let value = self.call_candidate(class, method_name, call, unboxed, &rest);
            let boxed = match value {
                Some(v) => self.box_dispatch_result(v, class, method_name, call)?,
                None => self.builder.build_const(IrValue::Null)?,
            };
            let from = self.builder.current_block()?;
            self.builder.add_phi_incoming(merge, result, from, boxed)?;
            self.builder.build_branch(merge)?;

            self.builder.switch_to_block(miss);
        }

        // Nothing matched: the throw never returns, the edge keeps the phi whole.
        let null = self.throw_unresolved_dynamic_call(no_match_message)?;
        let from = self.builder.current_block()?;
        self.builder.add_phi_incoming(merge, result, from, null)?;
        self.builder.build_branch(merge)?;

        self.builder.switch_to_block(merge);
        self.boxed_value_regs.insert(result);
        Some(result)
    }

    /// Call one candidate's runtime function on the unboxed receiver: a MIR
    /// wrapper through its declared signature, an extern with pointer slots.
    fn call_candidate(
        &mut self,
        class: &str,
        method_name: &str,
        call: &RuntimeFunctionCall,
        receiver: IrId,
        rest: &[(IrId, TypeId)],
    ) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let _ = (class, method_name);
        let runtime_name = call.runtime_name;
        if call.is_mir_wrapper {
            let (params, ret) = self
                .get_stdlib_mir_wrapper_signature(runtime_name)
                .unwrap_or_else(|| {
                    let mut p = vec![ptr_u8.clone()];
                    p.extend(rest.iter().map(|_| ptr_u8.clone()));
                    (p, ptr_u8.clone())
                });
            let mut arg_regs = vec![receiver];
            for (i, (reg, _)) in rest.iter().enumerate() {
                let actual = self.builder.get_register_type(*reg).unwrap_or(IrType::I64);
                let expected = params.get(i + 1).cloned().unwrap_or(actual.clone());
                arg_regs.push(self.maybe_box_for_extern_call(*reg, &actual, &expected)?);
            }
            let func = self.register_stdlib_mir_forward_ref(runtime_name, params, ret.clone());
            return self.builder.build_call_direct(func, arg_regs, ret);
        }
        let mut arg_regs = vec![receiver];
        for (reg, _) in rest {
            let actual = self.builder.get_register_type(*reg).unwrap_or(IrType::I64);
            arg_regs.push(self.maybe_box_for_extern_call(*reg, &actual, &ptr_u8)?);
        }
        let param_types: Vec<IrType> = arg_regs.iter().map(|_| ptr_u8.clone()).collect();
        let ret = if call.has_return {
            ptr_u8.clone()
        } else {
            IrType::Void
        };
        let func = self.get_or_register_extern_function(runtime_name, param_types, ret.clone());
        self.builder.build_call_direct(func, arg_regs, ret)
    }

    /// A candidate's result as a Dynamic box: a scalar by its kind, a String
    /// as a string box, an object with the class the mapping says the method
    /// returns (its own box tag, so the next call on it dispatches too).
    pub(crate) fn box_dispatch_result(
        &mut self,
        value: IrId,
        class: &str,
        method_name: &str,
        call: &RuntimeFunctionCall,
    ) -> Option<IrId> {
        use crate::stdlib::IrTypeDescriptor;
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        if !call.has_return {
            return self.builder.build_const(IrValue::Null);
        }
        // The mapping's declared return says what the pointer is; the
        // register type alone cannot tell a string from an object.
        let ty = match call.return_type.as_ref() {
            Some(IrTypeDescriptor::String | IrTypeDescriptor::PtrString) => IrType::String,
            _ => self.builder.get_register_type(value).unwrap_or(IrType::I64),
        };
        let boxed = self.box_dispatch_value(value, ty, class, method_name)?;
        // Recorded as a box so a binding does not take it for a raw handle.
        self.boxed_value_regs.insert(boxed);
        Some(boxed)
    }

    fn box_dispatch_value(
        &mut self,
        value: IrId,
        ty: IrType,
        class: &str,
        method_name: &str,
    ) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        match ty {
            IrType::Void => self.builder.build_const(IrValue::Null),
            IrType::I32 | IrType::I64 | IrType::U32 | IrType::U64 => {
                self.box_primitive_as_dynamic(value, ty, PrimBoxKind::Int)
            }
            IrType::F32 | IrType::F64 => {
                self.box_primitive_as_dynamic(value, ty, PrimBoxKind::Float)
            }
            IrType::Bool => self.box_primitive_as_dynamic(value, ty, PrimBoxKind::Bool),
            IrType::String => {
                let as_ptr = self.builder.build_bitcast(value, ptr_u8.clone())?;
                let box_fn = self.get_or_register_extern_function(
                    "haxe_box_haxestring_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(box_fn, vec![as_ptr], ptr_u8)
            }
            _ => {
                let returned_class = self
                    .stdlib_mapping
                    .class_key(class)
                    .and_then(|k| self.stdlib_mapping.return_class(k, method_name));
                let tag = returned_class.map_or(0, |c| self.stdlib_class_tag(c));
                let tag_reg = self.builder.build_const(IrValue::U32(tag))?;
                let as_ptr = self.builder.build_bitcast(value, ptr_u8.clone())?;
                let box_fn = self.get_or_register_extern_function(
                    "haxe_box_reference_ptr",
                    vec![ptr_u8.clone(), IrType::U32],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(box_fn, vec![as_ptr, tag_reg], ptr_u8)
            }
        }
    }
}
