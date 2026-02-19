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
use crate::type_system::{DynamicValue, TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TYPE_NULL, TYPE_STRING};

/// Haxe ValueType enum ordinals (matches Type.hx ValueType)
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
    // For now, we don't have function type IDs. Will be extended later.
    false
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

/// Type.typeof(v) -> ValueType
///
/// Returns the ValueType enum ordinal for a value.
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
        match dv.type_id {
            t if t == TYPE_NULL => TVALUETYPE_TNULL,
            t if t == TYPE_INT => TVALUETYPE_TINT,
            t if t == TYPE_FLOAT => TVALUETYPE_TFLOAT,
            t if t == TYPE_BOOL => TVALUETYPE_TBOOL,
            t if t == TYPE_STRING => TVALUETYPE_TCLASS, // String is a class in Haxe
            t if t == anon_object::TYPE_ANON_OBJECT => TVALUETYPE_TOBJECT,
            t if t.0 >= 1000 => TVALUETYPE_TCLASS, // User-defined types are classes
            _ => TVALUETYPE_TUNKNOWN,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::type_system::{haxe_box_float_ptr, haxe_box_int_ptr};

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
}
