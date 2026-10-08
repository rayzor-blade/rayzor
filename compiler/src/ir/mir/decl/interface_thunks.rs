//! Interface slots adapt an implementation to the interface's declared types.

use super::*;

impl<'a> HirToMirContext<'a> {
    pub(crate) fn interface_slot_signature(
        &self,
        interface: SymbolId,
        name: InternedString,
    ) -> Option<(Vec<TypeId>, TypeId)> {
        let mut pending = vec![interface];
        let mut seen = BTreeSet::new();
        while let Some(interface) = pending.pop() {
            if !seen.insert(interface) {
                continue;
            }
            for declaration in self.current_hir_types.values() {
                if let HirTypeDecl::Interface(declaration) = declaration {
                    if declaration.symbol_id == interface {
                        if let Some(method) = declaration.methods.iter().find(|m| m.name == name) {
                            return Some((
                                method.params.iter().map(|p| p.ty).collect(),
                                method.return_type,
                            ));
                        }
                    }
                }
            }
            let owner = self.class_qualified_name(interface)?;
            let qname = format!("{}.{}", owner, self.string_interner.get(name)?);
            if let Some(signature) = self.symbol_table.all_symbols().find_map(|symbol| {
                (symbol
                    .qualified_name
                    .and_then(|q| self.string_interner.get(q))
                    == Some(qname.as_str()))
                .then(|| self.resolve_function_type_signature(symbol.type_id))
                .flatten()
            }) {
                return Some(signature);
            }
            if let Some(ty) = self.interface_method_return_types.get(&(interface, name)) {
                let name = self.string_interner.get(name)?;
                if name.starts_with("get_") {
                    return Some((Vec::new(), *ty));
                }
                if name.starts_with("set_") {
                    return Some((vec![*ty], *ty));
                }
            }
            if let Some(parents) = self.interface_extends.get(&interface) {
                pending.extend(parents);
            }
        }
        None
    }

    pub(crate) fn prepare_interface_slot_args(
        &mut self,
        interface: SymbolId,
        name: InternedString,
        arguments: &mut [IrId],
        types: &[TypeId],
    ) -> Option<Vec<IrType>> {
        let formals = self
            .interface_slot_signature(interface, name)
            .map(|s| s.0)
            .unwrap_or_default();
        let mut params = vec![IrType::Ptr(Box::new(IrType::Void))];
        for (i, arg) in arguments.iter_mut().enumerate().skip(1) {
            if let (Some(source), Some(target)) = (types.get(i - 1), formals.get(i - 1)) {
                *arg = self.adapt_interface_slot_value(*arg, *source, *target)?;
            }
            params.push(self.builder.get_register_type(*arg).unwrap_or(IrType::I64));
        }
        Some(params)
    }

    pub(crate) fn interface_slot_return_type(
        &self,
        interface: SymbolId,
        name: InternedString,
        method: SymbolId,
        expression_type: TypeId,
    ) -> (IrType, Option<TypeId>) {
        if let Some((_, result)) = self.interface_slot_signature(interface, name) {
            let concrete = matches!(
                self.type_table
                    .get(self.resolve_through_aliases(result))
                    .map(|t| &t.kind),
                Some(
                    TypeKind::Int
                        | TypeKind::Float
                        | TypeKind::Bool
                        | TypeKind::String
                        | TypeKind::Void
                )
            ) || matches!(self.type_table.get(self.resolve_through_aliases(result)).map(|t| &t.kind),
                    Some(TypeKind::Interface { type_args, .. } | TypeKind::Class { type_args, .. }) if type_args.is_empty());
            if concrete {
                return (self.convert_type(result), Some(result));
            }
        }
        self.resolve_interface_method_return_type_full(method, expression_type)
    }

    fn interface_slot_needs_conversion(&self, source: TypeId, target: TypeId) -> bool {
        let source_iface = self.get_interface_symbol(source);
        let target_iface = self.get_interface_symbol(target);
        let source_class = self.get_class_symbol(self.resolve_through_aliases(source));
        let target_class = self.get_class_symbol(self.resolve_through_aliases(target));
        if source_iface != target_iface {
            if source_iface.is_some() && (target_iface.is_some() || target_class.is_some())
                || target_iface.is_some() && source_class.is_some()
            {
                return true;
            }
        }
        let numeric = |ty| matches!(ty, IrType::I32 | IrType::F32 | IrType::F64);
        let from = self.convert_type(source);
        let to = self.convert_type(target);
        from != to && numeric(from) && numeric(to)
    }

    pub(crate) fn adapt_interface_slot_value(
        &mut self,
        value: IrId,
        source: TypeId,
        target: TypeId,
    ) -> Option<IrId> {
        if !self.interface_slot_needs_conversion(source, target) {
            return Some(value);
        }
        let source_iface = self.get_interface_symbol(source);
        let target_iface = self.get_interface_symbol(target);
        let ptr = IrType::Ptr(Box::new(IrType::U8));
        let mut value = value;
        if source_iface.is_some() {
            let identity = self.get_or_register_extern_function(
                "haxe_iface_identity",
                vec![ptr.clone()],
                ptr.clone(),
            );
            value = self
                .builder
                .build_call_direct(identity, vec![value], ptr.clone())?;
        }
        if let Some(interface) = target_iface {
            let type_id = self.deterministic_iface_or_enum_type_id(interface, "iface")?;
            let type_id = self.builder.build_const(IrValue::I32(type_id as i32))?;
            let wrap = self.get_or_register_extern_function(
                "haxe_iface_fat_ptr_build",
                vec![ptr.clone(), IrType::I32],
                ptr.clone(),
            );
            value = self
                .builder
                .build_call_direct(wrap, vec![value, type_id], ptr)?;
        }
        let actual = self.builder.get_register_type(value)?;
        let expected = self.convert_type(target);
        if source_iface.is_some() || target_iface.is_some() {
            self.coerce_reg_to(value, &expected)
        } else {
            Some(self.reconcile_extern_return(value, &actual, &expected))
        }
    }

    pub(crate) fn interface_slot_thunk_name(
        &self,
        method: &str,
        interface: SymbolId,
    ) -> Option<String> {
        let interface = self.class_qualified_name(interface)?;
        let sanitize = |s: &str| {
            s.chars()
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect::<String>()
        };
        Some(format!(
            "__interface_dispatch_thunk__{}__{}",
            sanitize(&interface),
            sanitize(method)
        ))
    }

    pub(crate) fn ensure_interface_slot_thunk(
        &mut self,
        method_id: IrFunctionId,
        method_symbol: Option<SymbolId>,
        interface: SymbolId,
        name: InternedString,
    ) -> Option<IrFunctionId> {
        let method_symbol = method_symbol.or_else(|| {
            self.builder
                .module
                .functions
                .get(&method_id)
                .map(|f| f.symbol_id)
        })?;
        let symbol = self.symbol_table.get_symbol(method_symbol)?;
        let qname = self
            .string_interner
            .get(symbol.qualified_name.unwrap_or(symbol.name))?
            .to_owned();
        let thunk_name = self.interface_slot_thunk_name(&qname, interface)?;
        if let Some((id, _)) = self
            .builder
            .module
            .functions
            .iter()
            .find(|(_, f)| f.name == thunk_name)
        {
            return Some(*id);
        }
        let (native_params, native_return) =
            self.resolve_function_type_signature(symbol.type_id)?;
        let (slot_params, slot_return) = self.interface_slot_signature(interface, name)?;
        if native_params.len() != slot_params.len() {
            return None;
        }
        let method_sig = self
            .builder
            .module
            .functions
            .get(&method_id)
            .map(|f| f.signature.clone());
        let native_ir_params: Vec<_> = native_params
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                method_sig
                    .as_ref()
                    .and_then(|sig| sig.parameters.get(i + 1))
                    .map(|p| p.ty.clone())
                    .unwrap_or_else(|| self.convert_type(*ty))
            })
            .collect();
        let native_ir_return = method_sig
            .as_ref()
            .map(|sig| sig.return_type.clone())
            .unwrap_or_else(|| self.convert_type(native_return));
        let slot_ir_return = if self.interface_slot_needs_conversion(native_return, slot_return) {
            self.convert_type(slot_return)
        } else {
            native_ir_return.clone()
        };
        let mut sig = FunctionSignatureBuilder::new()
            .param("env".to_owned(), IrType::Ptr(Box::new(IrType::U8)))
            .param("this".to_owned(), IrType::Ptr(Box::new(IrType::Void)))
            .returns(slot_ir_return)
            .calling_convention(CallingConvention::Haxe);
        for (i, (slot, native)) in slot_params.iter().zip(&native_params).enumerate() {
            let ty = if self.interface_slot_needs_conversion(*slot, *native) {
                self.convert_type(*slot)
            } else {
                native_ir_params[i].clone()
            };
            sig = sig.param(format!("p{i}"), ty);
        }
        let wrapper_symbol = SymbolId::from_raw(u32::MAX - 3000 - self.next_wrapper_id);
        self.next_wrapper_id += 1;
        let saved_function = self.builder.current_function;
        let saved_block = self.builder.current_block;
        let saved_symbols = std::mem::take(&mut self.symbol_map);
        let saved_moves = std::mem::take(&mut self.strict_move_locals);
        let saved_results = std::mem::take(&mut self.interface_call_result_types);
        let saved_boxes = std::mem::take(&mut self.boxed_value_regs);
        self.reset_move_recorder();
        let id = self
            .builder
            .start_function(wrapper_symbol, thunk_name, sig.build());
        let emitted = (|| {
            let function = self.builder.current_function()?;
            let mut args = (1..native_params.len() + 2)
                .map(|i| function.get_param_reg(i))
                .collect::<Option<Vec<_>>>()?;
            for (i, (slot, native)) in slot_params.iter().zip(&native_params).enumerate() {
                args[i + 1] = self.adapt_interface_slot_value(args[i + 1], *slot, *native)?;
            }
            let result = self
                .builder
                .build_call_direct(method_id, args, native_ir_return.clone());
            let result = if native_ir_return == IrType::Void {
                None
            } else {
                Some(self.adapt_interface_slot_value(result?, native_return, slot_return)?)
            };
            self.builder.build_return(result)?;
            Some(())
        })();
        self.check_move_flow();
        self.builder.finish_function();
        self.builder.current_function = saved_function;
        self.builder.current_block = saved_block;
        self.symbol_map = saved_symbols;
        self.strict_move_locals = saved_moves;
        self.interface_call_result_types = saved_results;
        self.boxed_value_regs = saved_boxes;
        emitted.map(|()| id)
    }
}
