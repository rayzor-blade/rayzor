//! The erased-slot word of `runtime/src/erased_word.rs`, emitted as MIR
//! instructions for passes that run after lowering.
//!
//! With `RAYZOR_ERASED_WORDS` unset a slot holds the scalar's raw bits;
//! set, the self-describing word. `mir/helpers/words.rs` is the same
//! encoding through the builder.

use super::{
    BinaryOp, CompareOp, IrExternFunction, IrFunction, IrFunctionId, IrFunctionSignature, IrId,
    IrInstruction, IrModule, IrParameter, IrType, IrValue,
};

pub(crate) const INT_TAG: i64 = 0xFFFE_0000_0000_0000_u64 as i64;
pub(crate) const DOUBLE_OFFSET: i64 = 1 << 49;
pub(crate) const FALSE_WORD: i64 = 0x06;
pub(crate) const CANONICAL_NAN: i64 = 0x7FF8_0000_0000_0000;

/// A type-parameter tag fixup resolved to `tag`: a word tag keeps its marker
/// and carries `tag` as the element-kind hint in bits 8..15.
pub(crate) fn resolve_tag_fixup(current: &IrValue, tag: i32) -> IrValue {
    let word = rayzor_runtime::type_system::ValueTag::Word as i32;
    match current {
        IrValue::I32(old) if old & 0xff == word => IrValue::I32(word | (tag & 0xff) << 8),
        _ => IrValue::I32(tag),
    }
}

/// Whether `ty` is a scalar an erased slot holds encoded.
pub(crate) fn is_word_scalar(ty: &IrType) -> bool {
    matches!(ty, IrType::F64 | IrType::F32 | IrType::I32 | IrType::Bool)
}

/// The scalars `function` widened to an i64 slot without changing them: a
/// Float's bits, an Int or Bool extended. Keyed by the widened register.
pub(crate) fn widened_scalars(
    function: &IrFunction,
) -> std::collections::BTreeMap<IrId, (IrId, IrType)> {
    let mut out = std::collections::BTreeMap::new();
    for block in function.cfg.blocks.values() {
        for instruction in &block.instructions {
            let (dest, src, ty) = match instruction {
                IrInstruction::BitCast {
                    dest,
                    src,
                    ty: IrType::I64,
                } => match function.register_types.get(src) {
                    Some(ty @ (IrType::F64 | IrType::F32)) => (*dest, *src, ty.clone()),
                    _ => continue,
                },
                IrInstruction::Cast {
                    dest,
                    src,
                    from_ty: from @ (IrType::I32 | IrType::Bool),
                    to_ty: IrType::I64,
                } => (*dest, *src, from.clone()),
                _ => continue,
            };
            out.insert(dest, (src, ty));
        }
    }
    out
}

/// The runtime decoders, declared before a pass borrows a function.
#[derive(Clone, Copy)]
pub(crate) struct Decoders {
    to_float: IrFunctionId,
    to_bool: IrFunctionId,
    to_dynamic: IrFunctionId,
    from_dynamic: IrFunctionId,
}

impl Decoders {
    /// Declared only with `RAYZOR_ERASED_WORDS` set; unset, decoding needs
    /// no call.
    pub(crate) fn declare(module: &mut IrModule) -> Option<Self> {
        crate::debug_flags::erased_words().then(|| Self {
            to_float: extern_fn(module, "haxe_word_to_float", IrType::I64, IrType::F64),
            to_bool: extern_fn(module, "haxe_word_to_bool", IrType::I64, IrType::Bool),
            to_dynamic: extern_fn(
                module,
                "haxe_word_to_dynamic",
                IrType::I64,
                IrType::dynamic_ptr(),
            ),
            from_dynamic: extern_fn(
                module,
                "haxe_word_from_dynamic",
                IrType::dynamic_ptr(),
                IrType::I64,
            ),
        })
    }
}

pub(crate) fn extern_fn(
    module: &mut IrModule,
    name: &str,
    param: IrType,
    ret: IrType,
) -> IrFunctionId {
    if let Some(f) = module.extern_functions.values().find(|f| f.name == name) {
        return f.id;
    }
    let id = module.alloc_function_id();
    module.add_extern_function(IrExternFunction {
        id,
        name: name.to_string(),
        symbol_id: crate::tast::SymbolId::from_raw(0),
        signature: IrFunctionSignature {
            parameters: vec![IrParameter {
                name: String::new(),
                ty: param,
                reg: IrId(0),
                by_ref: false,
            }],
            return_type: ret,
            calling_convention: super::CallingConvention::C,
            can_throw: false,
            type_params: vec![],
            uses_sret: false,
        },
        source: "runtime".to_string(),
    });
    id
}

/// Fresh registers for a function whose `next_reg_id` may lag its
/// `register_types`.
pub(crate) struct Regs<'f> {
    function: &'f mut IrFunction,
}

impl<'f> Regs<'f> {
    pub(crate) fn new(function: &'f mut IrFunction) -> Self {
        let next = function
            .register_types
            .keys()
            .map(|reg| reg.0 + 1)
            .max()
            .unwrap_or(0)
            .max(function.next_reg_id);
        function.next_reg_id = next;
        Self { function }
    }

    pub(crate) fn function(&mut self) -> &mut IrFunction {
        self.function
    }

    pub(crate) fn fresh(&mut self, ty: IrType) -> IrId {
        let reg = self.function.alloc_reg();
        self.function.register_types.insert(reg, ty);
        reg
    }

    pub(crate) fn constant(&mut self, out: &mut Vec<IrInstruction>, value: i64) -> IrId {
        let dest = self.fresh(IrType::I64);
        out.push(IrInstruction::Const {
            dest,
            value: IrValue::I64(value),
        });
        dest
    }

    fn binop(
        &mut self,
        out: &mut Vec<IrInstruction>,
        op: BinaryOp,
        left: IrId,
        right: IrId,
    ) -> IrId {
        let dest = self.fresh(IrType::I64);
        out.push(IrInstruction::BinOp {
            dest,
            op,
            left,
            right,
        });
        dest
    }

    pub(crate) fn cast(
        &mut self,
        out: &mut Vec<IrInstruction>,
        src: IrId,
        from: &IrType,
        to: &IrType,
    ) -> IrId {
        let dest = self.fresh(to.clone());
        out.push(IrInstruction::Cast {
            dest,
            src,
            from_ty: from.clone(),
            to_ty: to.clone(),
        });
        dest
    }

    pub(crate) fn bitcast(&mut self, out: &mut Vec<IrInstruction>, src: IrId, to: &IrType) -> IrId {
        let dest = self.fresh(to.clone());
        out.push(IrInstruction::BitCast {
            dest,
            src,
            ty: to.clone(),
        });
        dest
    }

    /// A Dynamic `src` as an erased-slot word: a scalar box opens, a reference
    /// is itself.
    pub(crate) fn encode_dynamic(
        &mut self,
        decoders: Option<&Decoders>,
        out: &mut Vec<IrInstruction>,
        src: IrId,
    ) -> IrId {
        let Some(decoders) = decoders else {
            return self.cast(out, src, &IrType::dynamic_ptr(), &IrType::I64);
        };
        let word = self.fresh(IrType::I64);
        out.push(call(word, decoders.from_dynamic, src));
        word
    }

    /// `src: ty` as the raw i64 bits a native field holds.
    pub(crate) fn raw(&mut self, out: &mut Vec<IrInstruction>, src: IrId, ty: &IrType) -> IrId {
        match ty {
            IrType::F32 => {
                let wide = self.cast(out, src, &IrType::F32, &IrType::F64);
                self.bitcast(out, wide, &IrType::I64)
            }
            IrType::F64 => self.bitcast(out, src, &IrType::I64),
            IrType::I32 | IrType::Bool => self.cast(out, src, ty, &IrType::I64),
            _ if *ty == IrType::dynamic_ptr() => self.cast(out, src, ty, &IrType::I64),
            _ => src,
        }
    }

    /// The `ty` value whose raw bits an array slot holds.
    pub(crate) fn from_slot(
        &mut self,
        out: &mut Vec<IrInstruction>,
        bits: IrId,
        ty: &IrType,
    ) -> IrId {
        match ty {
            IrType::F64 => self.bitcast(out, bits, ty),
            IrType::F32 => {
                let wide = self.bitcast(out, bits, &IrType::F64);
                self.cast(out, wide, &IrType::F64, &IrType::F32)
            }
            IrType::I64 | IrType::TypeVar(_) => bits,
            _ => self.cast(out, bits, &IrType::I64, ty),
        }
    }

    /// `func_id(arg) -> ty` into a fresh register.
    pub(crate) fn call(
        &mut self,
        out: &mut Vec<IrInstruction>,
        func_id: IrFunctionId,
        arg: IrId,
        ty: &IrType,
    ) -> IrId {
        let dest = self.fresh(ty.clone());
        out.push(call(dest, func_id, arg));
        dest
    }

    /// `src: ty` as the raw bits an array slot holds.
    pub(crate) fn to_slot(&mut self, out: &mut Vec<IrInstruction>, src: IrId, ty: &IrType) -> IrId {
        match ty {
            IrType::F64 | IrType::F32 => self.raw(out, src, ty),
            IrType::NullInt => self.bitcast(out, src, &IrType::I64),
            IrType::I64 | IrType::TypeVar(_) => src,
            _ => self.cast(out, src, ty, &IrType::I64),
        }
    }

    /// `src: ty` as an erased-slot word, appending its instructions to `out`.
    /// Not a word scalar: `src` unchanged.
    pub(crate) fn encode(&mut self, out: &mut Vec<IrInstruction>, src: IrId, ty: &IrType) -> IrId {
        let words = crate::debug_flags::erased_words();
        match ty {
            IrType::F32 => {
                let wide = self.cast(out, src, &IrType::F32, &IrType::F64);
                self.encode(out, wide, &IrType::F64)
            }
            IrType::F64 if !words => self.bitcast(out, src, &IrType::I64),
            IrType::F64 => {
                let bits = self.bitcast(out, src, &IrType::I64);
                let nan = self.fresh(IrType::Bool);
                out.push(IrInstruction::Cmp {
                    dest: nan,
                    op: CompareOp::FUno,
                    left: src,
                    right: src,
                });
                let canonical = self.constant(out, CANONICAL_NAN);
                let picked = self.fresh(IrType::I64);
                out.push(IrInstruction::Select {
                    dest: picked,
                    condition: nan,
                    true_val: canonical,
                    false_val: bits,
                });
                let offset = self.constant(out, DOUBLE_OFFSET);
                self.binop(out, BinaryOp::Add, picked, offset)
            }
            IrType::I32 | IrType::Bool if !words => self.cast(out, src, ty, &IrType::I64),
            IrType::I32 => {
                let wide = self.cast(out, src, &IrType::I32, &IrType::I64);
                let low = self.constant(out, 0xFFFF_FFFF);
                let low = self.binop(out, BinaryOp::And, wide, low);
                let tag = self.constant(out, INT_TAG);
                self.binop(out, BinaryOp::Or, low, tag)
            }
            IrType::Bool => {
                let wide = self.cast(out, src, &IrType::Bool, &IrType::I64);
                let base = self.constant(out, FALSE_WORD);
                self.binop(out, BinaryOp::Add, wide, base)
            }
            _ => src,
        }
    }

    /// The `ty` scalar held by erased-slot `word`, appending its instructions
    /// to `out`. Not a word scalar: `word` unchanged.
    pub(crate) fn decode(
        &mut self,
        decoders: Option<&Decoders>,
        out: &mut Vec<IrInstruction>,
        word: IrId,
        ty: &IrType,
    ) -> IrId {
        let words = crate::debug_flags::erased_words();
        match ty {
            IrType::F64 if !words => self.bitcast(out, word, ty),
            IrType::F32 if !words => {
                let wide = self.bitcast(out, word, &IrType::F64);
                self.cast(out, wide, &IrType::F64, &IrType::F32)
            }
            // A word's low half is its Int.
            IrType::I32 | IrType::Bool if !words => self.cast(out, word, &IrType::I64, ty),
            IrType::I32 => self.cast(out, word, &IrType::I64, ty),
            IrType::F64 | IrType::F32 => {
                let func_id = decoders
                    .expect("declared with RAYZOR_ERASED_WORDS")
                    .to_float;
                let value = self.fresh(IrType::F64);
                out.push(call(value, func_id, word));
                if *ty == IrType::F32 {
                    self.cast(out, value, &IrType::F64, &IrType::F32)
                } else {
                    value
                }
            }
            // A scalar word boxes; a reference is its own Dynamic.
            _ if *ty == IrType::dynamic_ptr() && words => {
                let func_id = decoders
                    .expect("declared with RAYZOR_ERASED_WORDS")
                    .to_dynamic;
                let value = self.fresh(ty.clone());
                out.push(call(value, func_id, word));
                value
            }
            _ if *ty == IrType::dynamic_ptr() => self.cast(out, word, &IrType::I64, ty),
            IrType::Bool => {
                let func_id = decoders.expect("declared with RAYZOR_ERASED_WORDS").to_bool;
                let value = self.fresh(IrType::Bool);
                out.push(call(value, func_id, word));
                value
            }
            _ => word,
        }
    }
}

fn call(dest: IrId, func_id: IrFunctionId, arg: IrId) -> IrInstruction {
    IrInstruction::CallDirect {
        dest: Some(dest),
        func_id,
        args: vec![arg],
        arg_ownership: vec![super::OwnershipMode::Copy],
        type_args: vec![],
        is_tail_call: false,
    }
}
