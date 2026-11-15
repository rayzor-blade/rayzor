//! The word an erased generic slot holds: one 64-bit value that says what it
//! is, so every reader decodes it the same way whatever static type it has.
//!
//! - reference (object, String, Array, closure, enum cell, Dynamic box): the
//!   pointer itself, top 16 bits 0; null is 0.
//! - Int: `INT_TAG | (u32)i`.
//! - Bool: `FALSE_WORD` / `TRUE_WORD`, below the null page, never a pointer.
//! - Float: the f64 bits plus `DOUBLE_OFFSET`, NaNs canonical, so no Float
//!   word has top bits 0x0000 or 0xFFFE. A Single is stored as its exact f64.

use crate::type_system::{
    TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TypeId, dynamic_box_at, haxe_box_bool_ptr, haxe_box_float_ptr,
    haxe_box_int_ptr,
};

pub const INT_TAG: u64 = 0xFFFE_0000_0000_0000;
pub const DOUBLE_OFFSET: u64 = 1 << 49;
pub const FALSE_WORD: u64 = 0x06;
pub const TRUE_WORD: u64 = 0x07;
const TOP: u64 = 0xFFFF_0000_0000_0000;
const CANONICAL_NAN: u64 = 0x7FF8_0000_0000_0000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordKind {
    Reference,
    Int,
    Bool,
    Float,
}

pub fn kind(word: u64) -> WordKind {
    match word & TOP {
        0 if word == FALSE_WORD || word == TRUE_WORD => WordKind::Bool,
        0 => WordKind::Reference,
        INT_TAG => WordKind::Int,
        _ => WordKind::Float,
    }
}

pub fn encode_int(value: i32) -> u64 {
    INT_TAG | value as u32 as u64
}

pub fn encode_float(value: f64) -> u64 {
    let bits = if value.is_nan() {
        CANONICAL_NAN
    } else {
        value.to_bits()
    };
    bits.wrapping_add(DOUBLE_OFFSET)
}

pub fn encode_bool(value: bool) -> u64 {
    if value { TRUE_WORD } else { FALSE_WORD }
}

fn decode_float(word: u64) -> f64 {
    f64::from_bits(word.wrapping_sub(DOUBLE_OFFSET))
}

/// A scalar in a Dynamic box, if `p` is one.
fn boxed_scalar(p: *mut u8) -> Option<(TypeId, *mut u8)> {
    if p.is_null() {
        return None;
    }
    let dynamic = dynamic_box_at(p)?;
    [TYPE_INT, TYPE_FLOAT, TYPE_BOOL]
        .contains(&dynamic.type_id)
        .then_some((dynamic.type_id, dynamic.value_ptr))
}

/// The word for a Dynamic value: a boxed scalar is opened, anything else is
/// the reference itself.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_word_from_dynamic(value: *mut u8) -> u64 {
    match boxed_scalar(value) {
        Some((ty, payload)) if ty == TYPE_INT => {
            encode_int(unsafe { *(payload as *const i64) } as i32)
        }
        Some((ty, payload)) if ty == TYPE_FLOAT => {
            encode_float(unsafe { *(payload as *const f64) })
        }
        Some((_, payload)) => encode_bool(unsafe { *(payload as *const bool) }),
        None => value as u64,
    }
}

/// The Dynamic value for a word: a scalar is boxed, a reference is itself.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_word_to_dynamic(word: u64) -> *mut u8 {
    match kind(word) {
        WordKind::Reference => word as *mut u8,
        WordKind::Int => haxe_box_int_ptr(word as u32 as i32 as i64),
        WordKind::Bool => haxe_box_bool_ptr(word == TRUE_WORD),
        WordKind::Float => haxe_box_float_ptr(decode_float(word)),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_word_from_int(value: i32) -> u64 {
    encode_int(value)
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_word_from_float(value: f64) -> u64 {
    encode_float(value)
}

#[unsafe(no_mangle)]
pub extern "C" fn haxe_word_from_bool(value: bool) -> u64 {
    encode_bool(value)
}

/// The Int a word holds; a Float truncates, a boxed Int opens, null is 0.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_word_to_int(word: u64) -> i32 {
    match kind(word) {
        WordKind::Int => word as u32 as i32,
        WordKind::Float => decode_float(word) as i32,
        WordKind::Bool => (word == TRUE_WORD) as i32,
        WordKind::Reference => match haxe_word_from_dynamic(word as *mut u8) {
            inner if inner != word => haxe_word_to_int(inner),
            _ => 0,
        },
    }
}

/// The Float a word holds; an Int widens, a boxed scalar opens, null is 0.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_word_to_float(word: u64) -> f64 {
    match kind(word) {
        WordKind::Int => word as u32 as i32 as f64,
        WordKind::Float => decode_float(word),
        WordKind::Bool => (word == TRUE_WORD) as i32 as f64,
        WordKind::Reference => match haxe_word_from_dynamic(word as *mut u8) {
            inner if inner != word => haxe_word_to_float(inner),
            _ => 0.0,
        },
    }
}

/// The Bool a word holds; a boxed Bool opens, anything else is false.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_word_to_bool(word: u64) -> bool {
    match kind(word) {
        WordKind::Bool => word == TRUE_WORD,
        WordKind::Reference => match haxe_word_from_dynamic(word as *mut u8) {
            inner if inner != word => haxe_word_to_bool(inner),
            _ => false,
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ints_round_trip_and_are_ints() {
        for v in [0, 1, -1, 15, i32::MIN, i32::MAX] {
            let w = encode_int(v);
            assert_eq!(kind(w), WordKind::Int);
            assert_eq!(haxe_word_to_int(w), v);
            assert_eq!(haxe_word_to_float(w), v as f64);
        }
    }

    #[test]
    fn floats_round_trip_and_are_floats() {
        for v in [
            0.0,
            -0.0,
            0.65,
            -1.5,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let w = encode_float(v);
            assert_eq!(kind(w), WordKind::Float, "{v}");
            assert_eq!(haxe_word_to_float(w).to_bits(), v.to_bits(), "{v}");
        }
        let nan = encode_float(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF));
        assert_eq!(kind(nan), WordKind::Float);
        assert!(haxe_word_to_float(nan).is_nan());
    }

    #[test]
    fn a_single_survives_as_its_f64() {
        let s = 0.3f32;
        let w = encode_float(s as f64);
        assert_eq!(haxe_word_to_float(w) as f32, s);
    }

    #[test]
    fn bools_and_null_are_distinct() {
        assert_eq!(kind(encode_bool(true)), WordKind::Bool);
        assert_eq!(kind(encode_bool(false)), WordKind::Bool);
        assert!(haxe_word_to_bool(encode_bool(true)));
        assert!(!haxe_word_to_bool(encode_bool(false)));
        assert_eq!(kind(0), WordKind::Reference);
        assert_ne!(encode_float(0.0), 0);
        assert_ne!(encode_int(0), 0);
    }

    #[test]
    fn references_pass_through() {
        let boxed = Box::into_raw(Box::new(7u64)) as *mut u8;
        let w = boxed as u64;
        assert_eq!(kind(w), WordKind::Reference);
        assert_eq!(haxe_word_to_dynamic(w), boxed);
        unsafe { drop(Box::from_raw(boxed as *mut u64)) };
    }

    #[test]
    fn dynamic_boxes_open_and_rebox() {
        let w = haxe_word_from_dynamic(haxe_box_float_ptr(0.65));
        assert_eq!(kind(w), WordKind::Float);
        assert_eq!(haxe_word_to_float(w), 0.65);
        let w = haxe_word_from_dynamic(haxe_box_int_ptr(-3));
        assert_eq!(haxe_word_to_int(w), -3);
        let back = haxe_word_to_dynamic(encode_int(42));
        assert_eq!(haxe_word_to_int(haxe_word_from_dynamic(back)), 42);
    }
}
