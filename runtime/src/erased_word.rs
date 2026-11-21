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

/// `RAYZOR_UNBOXED_DYNAMIC=1` keeps an Int or Float word unboxed where it
/// becomes a Dynamic. Under test: readers that do not decode words yet fail.
fn unboxed_dynamic_scalars() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RAYZOR_UNBOXED_DYNAMIC").is_some_and(|v| v == "1"))
}

/// A scalar a Dynamic carries as its erased word instead of a box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Int(i32),
    Float(f64),
}

/// The scalar a Dynamic slot holds as its word: an Int or a Float word. A
/// sign-extended raw integer (top bits 0xFFFF) is none: no Float word has
/// them. Bool words are left out: 6 and 7 are also small raw integers.
pub fn scalar_word(p: *const u8) -> Option<Scalar> {
    let word = p as u64;
    if word & TOP == TOP {
        return None;
    }
    match kind(word) {
        WordKind::Int => Some(Scalar::Int(word as u32 as i32)),
        WordKind::Float => Some(Scalar::Float(decode_float(word))),
        WordKind::Reference | WordKind::Bool => None,
    }
}

/// A scalar word seen as a Dynamic box for the length of one reader call.
/// The box and its payload live on the caller's stack, so nothing is
/// allocated and nothing needs freeing; a reader must not keep the pointer.
#[repr(C)]
pub(crate) struct WordBox {
    type_id: u32,
    // Zero: a stack box carries no erased or reference box mark.
    mark: u32,
    value_ptr: *mut u8,
    payload: u64,
}

impl WordBox {
    /// The box for `p` when it is an Int or Float word.
    pub(crate) fn of(p: *const u8) -> Option<Self> {
        let (type_id, payload) = match scalar_word(p)? {
            Scalar::Int(i) => (TYPE_INT, i as i64 as u64),
            Scalar::Float(f) => (TYPE_FLOAT, f.to_bits()),
        };
        Some(Self {
            type_id: type_id.0,
            mark: 0,
            value_ptr: std::ptr::null_mut(),
            payload,
        })
    }

    /// The box's address, as a Dynamic reader takes it.
    pub(crate) fn ptr(&mut self) -> *mut u8 {
        self.value_ptr = &mut self.payload as *mut u64 as *mut u8;
        self as *mut Self as *mut u8
    }
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
    // An Int or Float word is its own Dynamic, unboxed, under the switch.
    if unboxed_dynamic_scalars() && scalar_word(word as *const u8).is_some() {
        return word as *mut u8;
    }
    match kind(word) {
        WordKind::Reference => word as *mut u8,
        WordKind::Int => haxe_box_int_ptr(word as u32 as i32 as i64),
        WordKind::Bool => haxe_box_bool_ptr(word == TRUE_WORD),
        WordKind::Float => haxe_box_float_ptr(decode_float(word)),
    }
}

/// The reference a word holds for a value typed as a class, array or
/// structure: a reference's Dynamic box opens to what it boxes, so an erased
/// slot filled through a Dynamic view reads as the reference itself.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_word_open_reference(word: u64) -> *mut u8 {
    let p = word as *mut u8;
    if !crate::type_system::is_reference_box(p) {
        return p;
    }
    match crate::type_system::dynamic_box_at(p) {
        Some(boxed) if !boxed.value_ptr.is_null() => boxed.value_ptr,
        _ => p,
    }
}

/// The interface value a word holds for an interface-typed parameter: a
/// class instance's Dynamic box becomes the interface's handle to it, a cell
/// holding the instance, as an upcast builds one.
#[unsafe(no_mangle)]
pub extern "C" fn haxe_word_open_interface(word: u64) -> *mut u8 {
    let p = word as *mut u8;
    match crate::type_system::class_box_instance(p) {
        Some(instance) => unsafe {
            let cell = libc::malloc(8) as *mut *mut u8;
            *cell = instance;
            cell as *mut u8
        },
        None => p,
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

/// Whether a value tag marks an erased-slot word. Bits 8..15 may carry the
/// instantiation's element tag as a hint.
pub fn is_word_tag(tag: i32) -> bool {
    tag & 0xff == crate::type_system::ValueTag::Word as i32
}

/// `(value, tag)` with a word opened into the native value and the tag it
/// needs: a scalar by its own kind, a reference by the hint, or unresolved
/// when there is none. Any other tag passes through.
pub fn native_value(value: i64, tag: i32) -> (i64, i32) {
    use crate::type_system::ValueTag;
    if !is_word_tag(tag) {
        return (value, tag);
    }
    if is_raw_hint(tag) {
        return (value, element_tag(tag));
    }
    let word = value as u64;
    match kind(word) {
        WordKind::Int => (i64::from(word as u32 as i32), ValueTag::Int as i32),
        WordKind::Float => (decode_float(word).to_bits() as i64, ValueTag::Float as i32),
        WordKind::Bool => (i64::from(word == TRUE_WORD), ValueTag::Bool as i32),
        // Null is null whatever the hint names.
        WordKind::Reference if word == 0 => (0, ValueTag::Dynamic as i32),
        WordKind::Reference => (value, (tag >> 8) & 0xff),
    }
}

/// Whether a word tag's hint names a binding whose slot holds the raw
/// value: `Null<Int>` and `Int64`, whose bits use the whole word.
pub fn is_raw_hint(tag: i32) -> bool {
    use crate::type_system::ValueTag;
    is_word_tag(tag)
        && matches!(
            ValueTag::from_i32(element_tag(tag)),
            Some(ValueTag::NullInt | ValueTag::Int64)
        )
}

/// The element type a tag describes: a word tag's hint, else the tag.
pub fn element_tag(tag: i32) -> i32 {
    if is_word_tag(tag) {
        (tag >> 8) & 0xff
    } else {
        tag
    }
}

/// The word for a native `value` described by `tag`; a reference is itself.
pub fn word_from_native(value: i64, tag: i32) -> i64 {
    use crate::type_system::ValueTag;
    (match ValueTag::from_i32(tag & 0xff) {
        Some(ValueTag::Int) => encode_int(value as i32),
        Some(ValueTag::Float) => encode_float(f64::from_bits(value as u64)),
        Some(ValueTag::Bool) => encode_bool(value != 0),
        _ => value as u64,
    }) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_compare_mixes_numbers_and_keeps_null_apart() {
        use crate::reflect::haxe_reflect_compare_typed as compare;
        use crate::type_system::ValueTag;
        let word = ValueTag::Word as i32;
        let (two, two_f) = (encode_int(2) as i64, encode_float(2.0) as i64);
        assert_eq!(compare(two_f, two, word), 0);
        assert_eq!(compare(encode_float(2.5) as i64, two, word), 1);
        let zero = encode_float(0.0) as i64;
        assert_ne!(compare(zero, 0, word), 0);
        assert_ne!(compare(0, zero, word), 0);
        assert_eq!(compare(0, 0, word), 0);
        // A `Null<Int>` slot's raw 0 is the Int 0.
        let null_int = word | (ValueTag::NullInt as i32) << 8;
        assert_eq!(compare(0, 0, null_int), 0);
    }

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
