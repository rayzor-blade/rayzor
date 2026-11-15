//! Moving a scalar into and out of an erased generic slot.
//!
//! With `RAYZOR_ERASED_WORDS` unset a slot holds the scalar's raw bits, as
//! it always has; set, it holds the self-describing word of
//! `runtime/src/erased_word.rs`. Every crossing goes through these two
//! helpers, so the switch changes every endpoint at once.

use super::*;

use crate::ir::erased_word::{self, CANONICAL_NAN, DOUBLE_OFFSET, FALSE_WORD, INT_TAG};

impl<'a> HirToMirContext<'a> {
    /// Whether `field` is declared at a type parameter: an erased slot.
    pub(crate) fn field_is_erased(&self, field: SymbolId) -> bool {
        let ty = self.symbol_table.get_symbol(field).map(|s| s.type_id);
        matches!(
            ty.and_then(|ty| self.type_table.get(ty)).map(|t| &t.kind),
            Some(TypeKind::TypeParameter { .. })
        )
    }

    /// The `want` scalar an erased field's loaded `word` holds, under
    /// `RAYZOR_ERASED_WORDS`; None when the read keeps its raw conversion.
    pub(crate) fn erased_field_word(&mut self, word: IrId, want: &IrType) -> Option<IrId> {
        if !crate::debug_flags::erased_words() {
            return None;
        }
        if *want == IrType::dynamic_ptr()
            || matches!(want, IrType::Ptr(inner) if **inner == IrType::Void)
        {
            // A scalar word boxes; a reference is its own Dynamic.
            let box_fn = self.get_or_register_extern_function(
                "haxe_word_to_dynamic",
                vec![IrType::I64],
                want.clone(),
            );
            return self
                .builder
                .build_call_direct(box_fn, vec![word], want.clone());
        }
        erased_word::is_word_scalar(want)
            .then(|| self.from_word(word, want))
            .flatten()
    }

    /// `value` as the i64 an erased slot holds. A register that already is a
    /// slot (i64, pointer, NullInt) is returned unchanged.
    pub(crate) fn into_word(&mut self, value: IrId) -> Option<IrId> {
        let ty = self.builder.get_register_type(value)?;
        let words = crate::debug_flags::erased_words();
        match ty {
            IrType::F64 | IrType::F32 if !words => self.builder.build_bitcast(value, IrType::I64),
            IrType::F32 => {
                let wide = self.builder.build_cast(value, IrType::F32, IrType::F64)?;
                self.into_word(wide)
            }
            IrType::F64 => {
                let bits = self.builder.build_bitcast(value, IrType::I64)?;
                let nan = self.builder.build_cmp(CompareOp::FUno, value, value)?;
                let canonical = self.builder.build_const(IrValue::I64(CANONICAL_NAN))?;
                let bits = self.builder.build_select(nan, canonical, bits)?;
                let offset = self.builder.build_const(IrValue::I64(DOUBLE_OFFSET))?;
                self.builder.build_binop(BinaryOp::Add, bits, offset)
            }
            IrType::I32 | IrType::Bool if !words => self.builder.build_cast(value, ty, IrType::I64),
            IrType::I32 => {
                let wide = self.builder.build_cast(value, IrType::I32, IrType::I64)?;
                let low = self.builder.build_const(IrValue::I64(0xFFFF_FFFF))?;
                let low = self.builder.build_binop(BinaryOp::And, wide, low)?;
                let tag = self.builder.build_const(IrValue::I64(INT_TAG))?;
                self.builder.build_binop(BinaryOp::Or, low, tag)
            }
            IrType::Bool => {
                let wide = self.builder.build_cast(value, IrType::Bool, IrType::I64)?;
                let base = self.builder.build_const(IrValue::I64(FALSE_WORD))?;
                self.builder.build_binop(BinaryOp::Add, wide, base)
            }
            _ => Some(value),
        }
    }

    /// The `ty` scalar an erased slot's `word` holds. Any other `ty` gets the
    /// word unchanged.
    pub(crate) fn from_word(&mut self, word: IrId, ty: &IrType) -> Option<IrId> {
        let words = crate::debug_flags::erased_words();
        match ty {
            IrType::F64 | IrType::F32 if !words => self.builder.build_bitcast(word, ty.clone()),
            IrType::I32 | IrType::Bool if !words => {
                self.builder.build_cast(word, IrType::I64, ty.clone())
            }
            // A word's low half is its Int.
            IrType::I32 => self.builder.build_cast(word, IrType::I64, IrType::I32),
            IrType::F64 | IrType::F32 => {
                let decode = self.get_or_register_extern_function(
                    "haxe_word_to_float",
                    vec![IrType::I64],
                    IrType::F64,
                );
                let value = self
                    .builder
                    .build_call_direct(decode, vec![word], IrType::F64)?;
                if *ty == IrType::F32 {
                    self.builder.build_cast(value, IrType::F64, IrType::F32)
                } else {
                    Some(value)
                }
            }
            IrType::Bool => {
                let decode = self.get_or_register_extern_function(
                    "haxe_word_to_bool",
                    vec![IrType::I64],
                    IrType::Bool,
                );
                self.builder
                    .build_call_direct(decode, vec![word], IrType::Bool)
            }
            _ => Some(word),
        }
    }
}

impl<'a> HirToMirContext<'a> {
    /// Encode every scalar a direct call passes to a parameter declared as a
    /// type parameter. Specialization keeps such a parameter an erased slot,
    /// so the caller is where the scalar's type is known.
    pub(crate) fn encode_erased_call_args(&self, module: &mut IrModule) {
        let erased: BTreeMap<IrFunctionId, Vec<bool>> = self
            .function_param_hir_types
            .iter()
            .map(|(id, params)| {
                let flags: Vec<bool> = params
                    .iter()
                    .map(|ty| {
                        matches!(
                            self.type_table.get(*ty).map(|t| &t.kind),
                            Some(TypeKind::TypeParameter { .. })
                        )
                    })
                    .collect();
                (*id, flags)
            })
            .filter(|(id, flags)| {
                flags.iter().any(|erased| *erased)
                    && module.functions.get(id).is_none_or(|f| {
                        // Declared params trail the receiver.
                        let ir = &f.signature.parameters;
                        ir.len() >= flags.len()
                            && ir[ir.len() - flags.len()..]
                                .iter()
                                .zip(flags)
                                .all(|(p, erased)| !erased || p.ty == IrType::I64)
                    })
            })
            .collect();
        if erased.is_empty() {
            return;
        }
        for function in module.functions.values_mut() {
            let calls = function.cfg.blocks.values().any(|b| {
                b.instructions.iter().any(|i| {
                    matches!(i, IrInstruction::CallDirect { func_id, .. } if erased.contains_key(func_id))
                })
            });
            if !calls {
                continue;
            }
            let mut regs = crate::ir::erased_word::Regs::new(function);
            let block_ids: Vec<_> = regs.function().cfg.blocks.keys().copied().collect();
            for id in block_ids {
                let Some(block) = regs.function().cfg.blocks.get_mut(&id) else {
                    continue;
                };
                let old = std::mem::take(&mut block.instructions);
                let mut out = Vec::with_capacity(old.len());
                for mut instruction in old {
                    if let IrInstruction::CallDirect { func_id, args, .. } = &mut instruction
                        && let Some(flags) = erased.get(func_id)
                        && let Some(offset) = args.len().checked_sub(flags.len())
                    {
                        for (arg, erased) in args[offset..].iter_mut().zip(flags) {
                            let ty = regs.function().register_types.get(arg).cloned();
                            if let Some(ty) =
                                ty.filter(|ty| *erased && erased_word::is_word_scalar(ty))
                            {
                                *arg = regs.encode(&mut out, *arg, &ty);
                            }
                        }
                    }
                    out.push(instruction);
                }
                if let Some(block) = regs.function().cfg.blocks.get_mut(&id) {
                    block.instructions = out;
                }
            }
        }
    }
}
