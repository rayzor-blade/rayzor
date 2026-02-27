//! Haxe Reflect and Type API runtime implementation
//!
//! Implements the core Reflect methods (hasField, field, setField, deleteField, fields,
//! isObject, isFunction, copy) and Type.typeof for anonymous objects.
//!
//! All functions receive raw `*mut u8` pointers from JIT code:
//! - `obj`: anonymous object handle (Box<Arc<AnonObject>>)
//! - `field`: HaxeString pointer containing the field name
//! - `value`: DynamicValue pointer for set operations

use crate::anon_object;
use crate::haxe_string::HaxeString;
use crate::type_system::{
    get_type_info, DynamicValue, TYPE_BOOL, TYPE_FLOAT, TYPE_FUNCTION, TYPE_INT, TYPE_NULL,
    TYPE_STRING,
};

/// Haxe ValueType constructor ordinals (matches Type.hx ValueType order)
pub const TVALUETYPE_TNULL: i32 = 0;
pub const TVALUETYPE_TINT: i32 = 1;
pub const TVALUETYPE_TFLOAT: i32 = 2;
pub const TVALUETYPE_TBOOL: i32 = 3;
pub const TVALUETYPE_TOBJECT: i32 = 4;
pub const TVALUETYPE_TFUNCTION: i32 = 5;
pub const TVALUETYPE_TCLASS: i32 = 6;
pub const TVALUETYPE_TENUM: i32 = 7;
pub const TVALUETYPE_TUNKNOWN: i32 = 8;

// ============================================================================
// Helper: extract field name bytes from HaxeString pointer
// ============================================================================

/// Extract (ptr, len) from a HaxeString pointer
///
/// # Safety
/// field_ptr must be a valid HaxeString pointer or null
unsafe fn extract_field_name(field_ptr: *mut u8) -> Option<(*const u8, u32)> {
    if field_ptr.is_null() {
        return None;
    }
    let hs = &*(field_ptr as *const HaxeString);
    if hs.ptr.is_null() || hs.len == 0 {
        return None;
    }
    Some((hs.ptr as *const u8, hs.len as u32))
}

// ============================================================================
// Reflect API
// ============================================================================

/// Reflect.hasField(obj, field) -> Bool
///
/// obj: anonymous object handle pointer
/// field: HaxeString pointer
#[no_mangle]
pub extern "C" fn haxe_reflect_has_field(obj: *mut u8, field: *mut u8) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe {
        if let Some((name_ptr, name_len)) = extract_field_name(field) {
            anon_object::rayzor_anon_has_field(obj, name_ptr, name_len)
        } else {
            false
        }
    }
}

/// Reflect.field(obj, field) -> Dynamic
///
/// obj: anonymous object handle pointer
/// field: HaxeString pointer
/// Returns: DynamicValue pointer (caller must manage), or null if field not found
#[no_mangle]
pub extern "C" fn haxe_reflect_field(obj: *mut u8, field: *mut u8) -> *mut u8 {
    if obj.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        if let Some((name_ptr, name_len)) = extract_field_name(field) {
            anon_object::rayzor_anon_get_field(obj, name_ptr, name_len)
        } else {
            std::ptr::null_mut()
        }
    }
}

/// Reflect.setField(obj, field, value) -> Void
///
/// obj: anonymous object handle pointer
/// field: HaxeString pointer
/// value: DynamicValue pointer
#[no_mangle]
pub extern "C" fn haxe_reflect_set_field(obj: *mut u8, field: *mut u8, value: *mut u8) {
    if obj.is_null() {
        return;
    }
    unsafe {
        if let Some((name_ptr, name_len)) = extract_field_name(field) {
            anon_object::rayzor_anon_set_field(obj, name_ptr, name_len, value);
        }
    }
}

/// Reflect.deleteField(obj, field) -> Bool
///
/// obj: anonymous object handle pointer
/// field: HaxeString pointer
/// Returns: true if field existed and was deleted
#[no_mangle]
pub extern "C" fn haxe_reflect_delete_field(obj: *mut u8, field: *mut u8) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe {
        if let Some((name_ptr, name_len)) = extract_field_name(field) {
            anon_object::rayzor_anon_delete_field(obj, name_ptr, name_len)
        } else {
            false
        }
    }
}

/// Reflect.fields(obj) -> Array<String>
///
/// obj: anonymous object handle pointer
/// Returns: HaxeArray pointer containing HaxeString pointers
#[no_mangle]
pub extern "C" fn haxe_reflect_fields(obj: *mut u8) -> *mut u8 {
    if obj.is_null() {
        return std::ptr::null_mut();
    }
    anon_object::rayzor_anon_fields(obj)
}

/// Reflect.isObject(v) -> Bool
///
/// Returns true if v is an anonymous object or class instance
/// v: DynamicValue pointer
#[no_mangle]
pub extern "C" fn haxe_reflect_is_object(v: *mut u8) -> bool {
    if v.is_null() {
        return false;
    }
    unsafe {
        let dv = *(v as *const DynamicValue);
        if dv.type_id == TYPE_FUNCTION {
            return false;
        }
        // Anonymous objects and user-defined types (classes) are "objects"
        dv.type_id == anon_object::TYPE_ANON_OBJECT || dv.type_id.0 >= 1000
    }
}

/// Reflect.isFunction(v) -> Bool
///
/// Returns true if v is a function/closure
/// v: DynamicValue pointer
#[no_mangle]
pub extern "C" fn haxe_reflect_is_function(v: *mut u8) -> bool {
    if v.is_null() {
        return false;
    }
    unsafe {
        let dv = *(v as *const DynamicValue);
        dv.type_id == TYPE_FUNCTION
    }
}

/// Reflect.copy(obj) -> Dynamic
///
/// Deep copies an anonymous object
/// obj: anonymous object handle pointer
/// Returns: new anonymous object handle pointer
#[no_mangle]
pub extern "C" fn haxe_reflect_copy(obj: *mut u8) -> *mut u8 {
    if obj.is_null() {
        return std::ptr::null_mut();
    }
    anon_object::rayzor_anon_copy(obj)
}

// ============================================================================
// Reflect.compare + Reflect.isEnumValue
// ============================================================================

/// Reflect.compare(a, b) -> Int
///
/// Compares two Dynamic values. Returns negative if a < b, 0 if equal, positive if a > b.
/// Both arguments are DynamicValue pointers (boxed values).
#[no_mangle]
pub extern "C" fn haxe_reflect_compare(a: *mut u8, b: *mut u8) -> i64 {
    if a.is_null() && b.is_null() {
        return 0;
    }
    if a.is_null() {
        return -1;
    }
    if b.is_null() {
        return 1;
    }
    unsafe {
        let dv_a = *(a as *const DynamicValue);
        let dv_b = *(b as *const DynamicValue);

        // Int × Int
        if dv_a.type_id == TYPE_INT && dv_b.type_id == TYPE_INT {
            let va = *(dv_a.value_ptr as *const i64);
            let vb = *(dv_b.value_ptr as *const i64);
            return (va - vb).signum();
        }

        // Float × Float
        if dv_a.type_id == TYPE_FLOAT && dv_b.type_id == TYPE_FLOAT {
            let va = *(dv_a.value_ptr as *const f64);
            let vb = *(dv_b.value_ptr as *const f64);
            return if va < vb {
                -1
            } else if va > vb {
                1
            } else {
                0
            };
        }

        // Int × Float or Float × Int
        if (dv_a.type_id == TYPE_INT && dv_b.type_id == TYPE_FLOAT)
            || (dv_a.type_id == TYPE_FLOAT && dv_b.type_id == TYPE_INT)
        {
            let fa = if dv_a.type_id == TYPE_FLOAT {
                *(dv_a.value_ptr as *const f64)
            } else {
                *(dv_a.value_ptr as *const i64) as f64
            };
            let fb = if dv_b.type_id == TYPE_FLOAT {
                *(dv_b.value_ptr as *const f64)
            } else {
                *(dv_b.value_ptr as *const i64) as f64
            };
            return if fa < fb {
                -1
            } else if fa > fb {
                1
            } else {
                0
            };
        }

        // String × String
        if dv_a.type_id == TYPE_STRING && dv_b.type_id == TYPE_STRING {
            let sa = &*(dv_a.value_ptr as *const crate::haxe_string::HaxeString);
            let sb = &*(dv_b.value_ptr as *const crate::haxe_string::HaxeString);
            let bytes_a = std::slice::from_raw_parts(sa.ptr, sa.len);
            let bytes_b = std::slice::from_raw_parts(sb.ptr, sb.len);
            return match bytes_a.cmp(bytes_b) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            };
        }

        // Bool × Bool
        if dv_a.type_id == TYPE_BOOL && dv_b.type_id == TYPE_BOOL {
            let va = *(dv_a.value_ptr as *const bool) as i64;
            let vb = *(dv_b.value_ptr as *const bool) as i64;
            return va - vb;
        }

        // Mismatched or unhandled types
        0
    }
}

/// Reflect.compare typed variant — compares raw type-erased i64 values.
///
/// Unlike haxe_reflect_compare (which expects DynamicValue* pointers), this function
/// accepts raw i64 values and interprets them based on a type_tag parameter.
/// This is used for generic code where values are type-erased to i64 and boxing
/// would require knowing the concrete type at compile time.
///
/// type_tag values: 1=Int, 2=Bool, 4=Float, 5=String
#[no_mangle]
pub extern "C" fn haxe_reflect_compare_typed(a: i64, b: i64, type_tag: i32) -> i64 {
    match type_tag {
        1 | 3 => {
            // Int comparison (type_tag 1=TYPE_INT, 3=legacy)
            (a - b).signum()
        }
        2 => {
            // Bool comparison
            let ba = (a != 0) as i64;
            let bb = (b != 0) as i64;
            ba - bb
        }
        4 => {
            // Float comparison — i64 bits reinterpreted as f64
            let fa = f64::from_bits(a as u64);
            let fb = f64::from_bits(b as u64);
            if fa < fb {
                -1
            } else if fa > fb {
                1
            } else {
                0
            }
        }
        5 => {
            // String comparison — i64 values are HaxeString pointers
            if a == 0 && b == 0 {
                return 0;
            }
            if a == 0 {
                return -1;
            }
            if b == 0 {
                return 1;
            }
            unsafe {
                let sa = &*(a as *const crate::haxe_string::HaxeString);
                let sb = &*(b as *const crate::haxe_string::HaxeString);
                let bytes_a = std::slice::from_raw_parts(sa.ptr, sa.len);
                let bytes_b = std::slice::from_raw_parts(sb.ptr, sb.len);
                match bytes_a.cmp(bytes_b) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                }
            }
        }
        _ => {
            // Unknown type: compare as raw i64
            (a - b).signum()
        }
    }
}

/// Reflect.isEnumValue(v) -> Bool
///
/// Returns true if v is an enum value (has enum_info in the type registry).
/// v: DynamicValue pointer
#[no_mangle]
pub extern "C" fn haxe_reflect_is_enum_value(v: *mut u8) -> bool {
    if v.is_null() {
        return false;
    }
    unsafe {
        let dv = *(v as *const DynamicValue);
        let registry = crate::type_system::TYPE_REGISTRY.read().unwrap();
        if let Some(ref map) = *registry {
            if let Some(info) = map.get(&dv.type_id) {
                return info.enum_info.is_some();
            }
        }
        false
    }
}

// ============================================================================
// Type API
// ============================================================================

/// Allocate a boxed enum payload with no constructor parameters.
/// Layout: [tag:i32][pad:i32] => returned as i64 pointer.
unsafe fn alloc_boxed_enum_tag_only(tag: i32) -> i64 {
    let ptr = libc::malloc(8) as *mut u8;
    if ptr.is_null() {
        return 0;
    }
    *(ptr as *mut i32) = tag;
    *((ptr as *mut i32).add(1)) = 0;
    ptr as i64
}

/// Allocate a boxed enum payload with one i64 constructor parameter.
/// Layout: [tag:i32][pad:i32][field0:i64] => returned as i64 pointer.
unsafe fn alloc_boxed_enum_with_i64(tag: i32, field0: i64) -> i64 {
    let ptr = libc::malloc(16) as *mut u8;
    if ptr.is_null() {
        return 0;
    }
    std::ptr::write_bytes(ptr, 0, 16);
    *(ptr as *mut i32) = tag;
    *(ptr.add(8) as *mut i64) = field0;
    ptr as i64
}

fn valuetype_tag_only(tag: i32) -> i64 {
    unsafe { alloc_boxed_enum_tag_only(tag) }
}

fn valuetype_tclass(type_id: i64) -> i64 {
    unsafe { alloc_boxed_enum_with_i64(TVALUETYPE_TCLASS, type_id) }
}

fn valuetype_tenum(type_id: i64) -> i64 {
    unsafe { alloc_boxed_enum_with_i64(TVALUETYPE_TENUM, type_id) }
}

/// Type.typeof(v) -> ValueType
///
/// Returns the ValueType ordinal for a value.
/// v: DynamicValue pointer
/// Returns: i32 ordinal (TNull=0, TInt=1, TFloat=2, TBool=3, TObject=4,
///          TFunction=5, TClass=6, TEnum=7, TUnknown=8)
#[no_mangle]
pub extern "C" fn haxe_type_typeof(v: *mut u8) -> i32 {
    if v.is_null() {
        return TVALUETYPE_TNULL;
    }
    unsafe {
        let dv = *(v as *const DynamicValue);
        if dv.type_id == TYPE_NULL {
            return TVALUETYPE_TNULL;
        }
        if dv.type_id == TYPE_INT {
            return TVALUETYPE_TINT;
        }
        if dv.type_id == TYPE_FLOAT {
            return TVALUETYPE_TFLOAT;
        }
        if dv.type_id == TYPE_BOOL {
            return TVALUETYPE_TBOOL;
        }
        if dv.type_id == TYPE_FUNCTION {
            return TVALUETYPE_TFUNCTION;
        }
        if dv.type_id == anon_object::TYPE_ANON_OBJECT {
            return TVALUETYPE_TOBJECT;
        }
        if dv.type_id == TYPE_STRING {
            // Haxe treats String as a class
            return TVALUETYPE_TCLASS;
        }

        if let Some(type_info) = get_type_info(dv.type_id) {
            if type_info.enum_info.is_some() {
                return TVALUETYPE_TENUM;
            }
            if type_info.class_info.is_some() {
                return TVALUETYPE_TCLASS;
            }
        }

        if dv.type_id.0 >= 1000 {
            // Unregistered user-defined runtime type: treat as class by convention.
            return TVALUETYPE_TCLASS;
        }

        TVALUETYPE_TUNKNOWN
    }
}

/// Build a boxed `ValueType` runtime enum value from the ordinal returned by
/// `haxe_type_typeof`.
///
/// This keeps the low-level typeof contract simple (`i32`) while allowing
/// compiler lowering to request a real enum value when needed.
#[no_mangle]
pub extern "C" fn haxe_type_typeof_value(v: *mut u8) -> i64 {
    let tag = haxe_type_typeof(v);
    match tag {
        TVALUETYPE_TCLASS => {
            let type_id = unsafe {
                if v.is_null() {
                    -1
                } else {
                    let dv = *(v as *const DynamicValue);
                    dv.type_id.0 as i64
                }
            };
            valuetype_tclass(type_id)
        }
        TVALUETYPE_TENUM => {
            let type_id = unsafe {
                if v.is_null() {
                    -1
                } else {
                    let dv = *(v as *const DynamicValue);
                    dv.type_id.0 as i64
                }
            };
            valuetype_tenum(type_id)
        }
        _ => valuetype_tag_only(tag),
    }
}

/// Trace a boxed ValueType enum returned by `haxe_type_typeof_value`.
///
/// This avoids relying on enum RTTI registration for `ValueType` and provides
/// stable output for `trace(Type.typeof(x))`.
#[no_mangle]
pub extern "C" fn haxe_trace_value_type(value: i64) {
    fn trace_text(s: &str) {
        crate::haxe_sys::haxe_trace_string(s.as_ptr(), s.len());
    }

    if value == 0 {
        trace_text("TNull");
        return;
    }

    let is_ptr_like = value > 0x1000;
    let tag = if is_ptr_like {
        unsafe { *(value as *const i32) }
    } else {
        value as i32
    };

    let payload = if is_ptr_like {
        unsafe { *((value as *const u8).add(8) as *const i64) }
    } else {
        0
    };

    match tag {
        TVALUETYPE_TNULL => trace_text("TNull"),
        TVALUETYPE_TINT => trace_text("TInt"),
        TVALUETYPE_TFLOAT => trace_text("TFloat"),
        TVALUETYPE_TBOOL => trace_text("TBool"),
        TVALUETYPE_TOBJECT => trace_text("TObject"),
        TVALUETYPE_TFUNCTION => trace_text("TFunction"),
        TVALUETYPE_TCLASS => {
            if payload >= 0 {
                if let Some(type_info) = get_type_info(crate::type_system::TypeId(payload as u32)) {
                    let s = format!("TClass({})", type_info.name);
                    trace_text(&s);
                } else {
                    let s = format!("TClass({})", payload);
                    trace_text(&s);
                }
            } else {
                trace_text("TClass");
            }
        }
        TVALUETYPE_TENUM => {
            if payload >= 0 {
                if let Some(type_info) = get_type_info(crate::type_system::TypeId(payload as u32)) {
                    let s = format!("TEnum({})", type_info.name);
                    trace_text(&s);
                } else {
                    let s = format!("TEnum({})", payload);
                    trace_text(&s);
                }
            } else {
                trace_text("TEnum");
            }
        }
        TVALUETYPE_TUNKNOWN => {
            let s = format!("TUnknown({})", payload);
            trace_text(&s);
        }
        _ => {
            let s = format!("<ValueType:{}>", tag);
            trace_text(&s);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::type_system::{haxe_box_float_ptr, haxe_box_function_ptr, haxe_box_int_ptr};

    unsafe extern "C" fn dummy_to_string(_ptr: *const u8) -> crate::type_system::StringPtr {
        crate::type_system::StringPtr {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    fn decode_valuetype_tag(v: i64) -> i32 {
        assert_ne!(v, 0, "ValueType should be boxed (non-null pointer)");
        unsafe { *(v as *const i32) }
    }

    fn decode_valuetype_payload(v: i64) -> i64 {
        assert_ne!(v, 0, "ValueType should be boxed (non-null pointer)");
        unsafe { *((v as *const u8).add(8) as *const i64) }
    }

    #[test]
    fn test_typeof_int() {
        let boxed = haxe_box_int_ptr(42);
        assert_eq!(haxe_type_typeof(boxed), TVALUETYPE_TINT);
        // Note: leaking for test simplicity
    }

    #[test]
    fn test_typeof_float() {
        let boxed = haxe_box_float_ptr(3.1);
        assert_eq!(haxe_type_typeof(boxed), TVALUETYPE_TFLOAT);
    }

    #[test]
    fn test_typeof_null() {
        assert_eq!(haxe_type_typeof(std::ptr::null_mut()), TVALUETYPE_TNULL);
    }

    #[test]
    fn test_reflect_is_function_true_for_boxed_function() {
        let closure_ptr = Box::into_raw(Box::new([0u8; 16])) as *mut u8;
        let boxed_fn = haxe_box_function_ptr(closure_ptr);
        assert!(haxe_reflect_is_function(boxed_fn));
        let vt = haxe_type_typeof_value(boxed_fn);
        assert_eq!(decode_valuetype_tag(vt), TVALUETYPE_TFUNCTION);
    }

    #[test]
    fn test_typeof_string_is_tclass_with_payload() {
        let mut hs = Box::new(crate::haxe_string::HaxeString {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        });
        crate::haxe_string::haxe_string_from_bytes(hs.as_mut(), b"x".as_ptr(), 1);
        let s = Box::into_raw(hs) as *mut u8;
        let dv = Box::into_raw(Box::new(DynamicValue {
            type_id: TYPE_STRING,
            value_ptr: s,
        })) as *mut u8;
        let vt = haxe_type_typeof_value(dv);
        assert_eq!(decode_valuetype_tag(vt), TVALUETYPE_TCLASS);
        assert_eq!(decode_valuetype_payload(vt), TYPE_STRING.0 as i64);
    }

    #[test]
    fn test_typeof_unknown_is_tunknown() {
        let fake = Box::into_raw(Box::new(DynamicValue {
            type_id: crate::type_system::TypeId(999),
            value_ptr: std::ptr::null_mut(),
        })) as *mut u8;
        let vt = haxe_type_typeof_value(fake);
        assert_eq!(decode_valuetype_tag(vt), TVALUETYPE_TUNKNOWN);
    }

    #[test]
    fn test_typeof_registered_user_type_is_tclass() {
        crate::type_system::register_type(
            crate::type_system::TypeId(4242),
            crate::type_system::TypeInfo {
                name: "UserClass",
                size: 8,
                align: 8,
                to_string: dummy_to_string,
                enum_info: None,
                class_info: Some(Box::leak(Box::new(crate::type_system::ClassInfo {
                    name: "UserClass",
                    super_type_id: None,
                    instance_fields: Box::leak(vec!["__type_id"].into_boxed_slice()),
                    static_fields: Box::leak(Vec::<&'static str>::new().into_boxed_slice()),
                }))),
            },
        );

        let dv = Box::into_raw(Box::new(DynamicValue {
            type_id: crate::type_system::TypeId(4242),
            value_ptr: std::ptr::null_mut(),
        })) as *mut u8;
        let vt = haxe_type_typeof_value(dv);
        assert_eq!(decode_valuetype_tag(vt), TVALUETYPE_TCLASS);
        assert_eq!(decode_valuetype_payload(vt), 4242);
    }

    #[test]
    fn test_typeof_value_primitive_int_not_tclass() {
        let boxed = haxe_box_int_ptr(7);
        let vt = haxe_type_typeof_value(boxed);
        assert_eq!(decode_valuetype_tag(vt), TVALUETYPE_TINT);
    }
}
