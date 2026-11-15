//! Moving a scalar into and out of an erased generic slot.
//!
//! With `RAYZOR_ERASED_WORDS` unset a slot holds the scalar's raw bits, as
//! it always has; set, it holds the self-describing word of
//! `runtime/src/erased_word.rs`. Every crossing goes through these two
//! helpers, so the switch changes every endpoint at once.

use super::*;

const INT_TAG: i64 = 0xFFFE_0000_0000_0000_u64 as i64;
const DOUBLE_OFFSET: i64 = 1 << 49;
const FALSE_WORD: i64 = 0x06;
const CANONICAL_NAN: i64 = 0x7FF8_0000_0000_0000;

impl<'a> HirToMirContext<'a> {
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
