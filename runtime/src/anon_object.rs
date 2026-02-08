//! Anonymous Object runtime for Haxe Dynamic/Any values
//!
//! Implements Arc-based refcounting with Copy-on-Write (COW) semantics.
//! Two storage modes:
//! - Inline: fixed-layout fields (compile-time known shape)
//! - Map: HashMap for runtime-flexible objects (Reflect.setField/deleteField)
//!
//! ## Handle Layout
//!
//! Each JIT variable holding an anon object stores a `*mut u8` that points to
//! a `Box<Arc<AnonObject>>`. This double indirection allows COW via `Arc::make_mut`
//! without changing the handle pointer seen by JIT code.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::type_system::{
    DynamicValue, TypeId, TYPE_BOOL, TYPE_FLOAT, TYPE_INT, TYPE_NULL, TYPE_STRING,
};

/// Type ID for anonymous objects in the DynamicValue type system
pub const TYPE_ANON_OBJECT: TypeId = TypeId(6);

/// Sentinel shape_id for Map-backed (dynamic) anonymous objects
pub const DYNAMIC_SHAPE: u32 = u32::MAX;

/// Anonymous object with Arc-based refcounting
#[derive(Clone)]
pub struct AnonObject {
    pub shape_id: u32,
    pub data: AnonData,
}

/// Storage for anonymous object fields
#[derive(Clone)]
pub enum AnonData {
    /// Fixed-layout: fields stored by index (sorted by field name)
    Inline(Vec<u64>),
    /// Dynamic: runtime-flexible field set (type_id, raw_value)
    Map(HashMap<String, (u32, u64)>),
}

/// Describes the field layout of an optimized anonymous object shape
#[derive(Clone, Debug)]
pub struct ShapeDescriptor {
    pub field_names: Vec<String>,
    pub field_types: Vec<u32>,
}

/// Global shape table
static SHAPE_TABLE: RwLock<Option<Vec<ShapeDescriptor>>> = RwLock::new(None);

fn ensure_shape_table() {
    let mut table = SHAPE_TABLE.write().unwrap();
    if table.is_none() {
        *table = Some(Vec::new());
    }
}

/// Register a new shape, returns shape_id
#[no_mangle]
pub extern "C" fn rayzor_register_shape(
    field_names_ptr: *const *const u8,
    field_name_lens_ptr: *const u32,
    field_types_ptr: *const u32,
    count: u32,
) -> u32 {
    ensure_shape_table();

    let mut field_names = Vec::with_capacity(count as usize);
    let mut field_types = Vec::with_capacity(count as usize);

    unsafe {
        for i in 0..count as usize {
            let name_ptr = *field_names_ptr.add(i);
            let name_len = *field_name_lens_ptr.add(i) as usize;
            let name_bytes = std::slice::from_raw_parts(name_ptr, name_len);
            let name = String::from_utf8_lossy(name_bytes).to_string();
            field_names.push(name);
            field_types.push(*field_types_ptr.add(i));
        }
    }

    let shape = ShapeDescriptor {
        field_names,
        field_types,
    };

    let mut table = SHAPE_TABLE.write().unwrap();
    let table = table.as_mut().unwrap();
    let shape_id = table.len() as u32;
    table.push(shape);
    shape_id
}

/// Ensure a shape is registered at the given shape_id.
///
/// descriptor_hs: HaxeString pointer containing "name1:type1,name2:type2,..."
/// (sorted alphabetically by name)
/// Type IDs: 0=Void, 1=Null, 2=Bool, 3=Int, 4=Float, 5=String
///
/// Idempotent: if shape_id is already registered, this is a no-op.
#[no_mangle]
pub extern "C" fn rayzor_ensure_shape(shape_id: u32, descriptor_hs: *mut u8) {
    ensure_shape_table();

    // Fast path: check if already registered (read lock only)
    {
        let table = SHAPE_TABLE.read().unwrap();
        if let Some(ref t) = *table {
            if (shape_id as usize) < t.len() && !t[shape_id as usize].field_names.is_empty() {
                return;
            }
        }
    }

    // Extract string data from HaxeString pointer
    if descriptor_hs.is_null() {
        return;
    }
    let hs = unsafe { &*(descriptor_hs as *const crate::haxe_string::HaxeString) };
    if hs.ptr.is_null() || hs.len == 0 {
        return;
    }
    let desc_str = unsafe {
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(hs.ptr as *const u8, hs.len))
    };

    let mut field_names = Vec::new();
    let mut field_types = Vec::new();

    for part in desc_str.split(',') {
        if part.is_empty() {
            continue;
        }
        if let Some((name, type_str)) = part.split_once(':') {
            field_names.push(name.to_string());
            field_types.push(type_str.parse::<u32>().unwrap_or(0));
        } else {
            field_names.push(part.to_string());
            field_types.push(0);
        }
    }

    let shape = ShapeDescriptor {
        field_names,
        field_types,
    };

    // Write lock to register
    let mut table = SHAPE_TABLE.write().unwrap();
    let table = table.as_mut().unwrap();

    // Grow table if needed
    while table.len() <= shape_id as usize {
        table.push(ShapeDescriptor {
            field_names: Vec::new(),
            field_types: Vec::new(),
        });
    }

    table[shape_id as usize] = shape;
}

/// Get shape descriptor by ID (internal helper)
fn get_shape(shape_id: u32) -> Option<ShapeDescriptor> {
    let table = SHAPE_TABLE.read().unwrap();
    table.as_ref()?.get(shape_id as usize).cloned()
}

// ============================================================================
// Handle helpers: Box<Arc<AnonObject>> stored as *mut u8
// ============================================================================

/// Borrow the Arc from a handle pointer (does NOT take ownership)
///
/// # Safety
/// ptr must be a valid handle returned by rayzor_anon_new or rayzor_anon_clone
unsafe fn borrow_arc(ptr: *mut u8) -> &'static Arc<AnonObject> {
    &*(ptr as *const Arc<AnonObject>)
}

/// Borrow the Arc mutably from a handle pointer (does NOT take ownership)
///
/// # Safety
/// ptr must be a valid handle, and no other references must exist
unsafe fn borrow_arc_mut(ptr: *mut u8) -> &'static mut Arc<AnonObject> {
    &mut *(ptr as *mut Arc<AnonObject>)
}

// ============================================================================
// Core API
// ============================================================================

/// Create a new anonymous object with the given shape
#[no_mangle]
pub extern "C" fn rayzor_anon_new(shape_id: u32, field_count: u32) -> *mut u8 {
    let data = if shape_id == DYNAMIC_SHAPE {
        AnonData::Map(HashMap::new())
    } else {
        AnonData::Inline(vec![0u64; field_count as usize])
    };

    let obj = AnonObject { shape_id, data };
    let arc = Arc::new(obj);
    let boxed = Box::new(arc);
    Box::into_raw(boxed) as *mut u8
}

/// Clone an anonymous object handle (creates new handle sharing the same Arc)
#[no_mangle]
pub extern "C" fn rayzor_anon_clone(ptr: *mut u8) -> *mut u8 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let arc_ref = borrow_arc(ptr);
        let cloned = Arc::clone(arc_ref);
        let boxed = Box::new(cloned);
        Box::into_raw(boxed) as *mut u8
    }
}

/// Drop an anonymous object handle (decrements Arc refcount, frees if zero)
#[no_mangle]
pub extern "C" fn rayzor_anon_drop(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _boxed: Box<Arc<AnonObject>> = Box::from_raw(ptr as *mut Arc<AnonObject>);
        // Box dropped here → Arc dropped → refcount decremented → object freed if zero
    }
}

/// Get field by index (optimized path for known shapes)
#[no_mangle]
pub extern "C" fn rayzor_anon_get_field_by_index(ptr: *mut u8, index: u32) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        let arc_ref = borrow_arc(ptr);
        match &arc_ref.data {
            AnonData::Inline(fields) => fields.get(index as usize).copied().unwrap_or(0),
            AnonData::Map(_) => 0,
        }
    }
}

/// Set field by index with COW (optimized path for known shapes)
#[no_mangle]
pub extern "C" fn rayzor_anon_set_field_by_index(ptr: *mut u8, index: u32, value: u64) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let arc = borrow_arc_mut(ptr);
        let obj = Arc::make_mut(arc);
        if let AnonData::Inline(fields) = &mut obj.data {
            if (index as usize) < fields.len() {
                fields[index as usize] = value;
            }
        }
    }
}

/// Check if field exists by name
#[no_mangle]
pub extern "C" fn rayzor_anon_has_field(ptr: *mut u8, name_ptr: *const u8, name_len: u32) -> bool {
    if ptr.is_null() || name_ptr.is_null() {
        return false;
    }
    unsafe {
        let arc_ref = borrow_arc(ptr);
        let name =
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(name_ptr, name_len as usize));

        eprintln!(
            "[DEBUG has_field] shape_id={}, name='{}', data_kind={}",
            arc_ref.shape_id,
            name,
            match &arc_ref.data {
                AnonData::Inline(f) => format!("Inline(len={})", f.len()),
                AnonData::Map(m) => format!("Map(len={})", m.len()),
            }
        );

        match &arc_ref.data {
            AnonData::Inline(_) => {
                if let Some(shape) = get_shape(arc_ref.shape_id) {
                    eprintln!("[DEBUG has_field] shape fields: {:?}", shape.field_names);
                    shape.field_names.iter().any(|n| n == name)
                } else {
                    eprintln!(
                        "[DEBUG has_field] get_shape({}) returned None!",
                        arc_ref.shape_id
                    );
                    false
                }
            }
            AnonData::Map(map) => map.contains_key(name),
        }
    }
}

/// Get field by name, returns boxed DynamicValue pointer (caller must free)
#[no_mangle]
pub extern "C" fn rayzor_anon_get_field(
    ptr: *mut u8,
    name_ptr: *const u8,
    name_len: u32,
) -> *mut u8 {
    if ptr.is_null() || name_ptr.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let arc_ref = borrow_arc(ptr);
        let name =
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(name_ptr, name_len as usize));

        match &arc_ref.data {
            AnonData::Inline(fields) => {
                if let Some(shape) = get_shape(arc_ref.shape_id) {
                    if let Some(idx) = shape.field_names.iter().position(|n| n == name) {
                        let value = fields[idx];
                        let type_id = shape.field_types[idx];
                        box_value_as_dynamic(type_id, value)
                    } else {
                        std::ptr::null_mut()
                    }
                } else {
                    std::ptr::null_mut()
                }
            }
            AnonData::Map(map) => {
                if let Some(&(type_id, value)) = map.get(name) {
                    box_value_as_dynamic(type_id, value)
                } else {
                    std::ptr::null_mut()
                }
            }
        }
    }
}

/// Set field by name with COW (dynamic path)
/// value_ptr: pointer to DynamicValue containing the value to store
#[no_mangle]
pub extern "C" fn rayzor_anon_set_field(
    ptr: *mut u8,
    name_ptr: *const u8,
    name_len: u32,
    value_ptr: *mut u8,
) {
    if ptr.is_null() || name_ptr.is_null() {
        return;
    }
    let name = unsafe {
        String::from_utf8_lossy(std::slice::from_raw_parts(name_ptr, name_len as usize)).to_string()
    };

    // Extract type_id and raw value from DynamicValue pointer
    let (type_id, raw_value) = if value_ptr.is_null() {
        (TYPE_NULL.0, 0u64)
    } else {
        unsafe {
            let dv = *(value_ptr as *const DynamicValue);
            let raw = if dv.value_ptr.is_null() {
                0u64
            } else {
                *(dv.value_ptr as *const u64)
            };
            (dv.type_id.0, raw)
        }
    };

    unsafe {
        let arc = borrow_arc_mut(ptr);
        let obj = Arc::make_mut(arc);

        match &mut obj.data {
            AnonData::Inline(fields) => {
                // Check if field exists in shape
                if let Some(shape) = get_shape(obj.shape_id) {
                    if let Some(idx) = shape.field_names.iter().position(|n| n == &name) {
                        fields[idx] = raw_value;
                        return;
                    }
                }
                // Field not in shape → promote to Map
                let mut map = HashMap::new();
                if let Some(shape) = get_shape(obj.shape_id) {
                    for (i, field_name) in shape.field_names.iter().enumerate() {
                        map.insert(field_name.clone(), (shape.field_types[i], fields[i]));
                    }
                }
                map.insert(name, (type_id, raw_value));
                obj.shape_id = DYNAMIC_SHAPE;
                obj.data = AnonData::Map(map);
            }
            AnonData::Map(map) => {
                map.insert(name, (type_id, raw_value));
            }
        }
    }
}

/// Delete field by name with COW (returns true if field existed)
#[no_mangle]
pub extern "C" fn rayzor_anon_delete_field(
    ptr: *mut u8,
    name_ptr: *const u8,
    name_len: u32,
) -> bool {
    if ptr.is_null() || name_ptr.is_null() {
        return false;
    }
    let name = unsafe {
        String::from_utf8_lossy(std::slice::from_raw_parts(name_ptr, name_len as usize)).to_string()
    };

    unsafe {
        let arc = borrow_arc_mut(ptr);
        let obj = Arc::make_mut(arc);

        match &mut obj.data {
            AnonData::Inline(fields) => {
                // Must promote to Map to delete a field
                if let Some(shape) = get_shape(obj.shape_id) {
                    let mut map = HashMap::new();
                    let mut found = false;
                    for (i, field_name) in shape.field_names.iter().enumerate() {
                        if field_name == &name {
                            found = true;
                            continue;
                        }
                        map.insert(field_name.clone(), (shape.field_types[i], fields[i]));
                    }
                    obj.shape_id = DYNAMIC_SHAPE;
                    obj.data = AnonData::Map(map);
                    found
                } else {
                    false
                }
            }
            AnonData::Map(map) => map.remove(&name).is_some(),
        }
    }
}

/// Get all field names as a HaxeArray of HaxeString pointers
#[no_mangle]
pub extern "C" fn rayzor_anon_fields(ptr: *mut u8) -> *mut u8 {
    use crate::haxe_array::HaxeArray;
    use crate::haxe_string::HaxeString;

    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let field_names: Vec<String> = unsafe {
        let arc_ref = borrow_arc(ptr);
        match &arc_ref.data {
            AnonData::Inline(_) => {
                if let Some(shape) = get_shape(arc_ref.shape_id) {
                    shape.field_names.clone()
                } else {
                    Vec::new()
                }
            }
            AnonData::Map(map) => {
                let mut keys: Vec<String> = map.keys().cloned().collect();
                keys.sort();
                keys
            }
        }
    };

    unsafe {
        let arr_layout = std::alloc::Layout::new::<HaxeArray>();
        let arr_ptr = std::alloc::alloc(arr_layout) as *mut HaxeArray;
        if arr_ptr.is_null() {
            return std::ptr::null_mut();
        }
        crate::haxe_array::haxe_array_new(arr_ptr, std::mem::size_of::<*mut HaxeString>());

        for name in &field_names {
            // Create a HaxeString for this field name
            let hs_layout = std::alloc::Layout::new::<HaxeString>();
            let hs_ptr = std::alloc::alloc(hs_layout) as *mut HaxeString;
            if !hs_ptr.is_null() {
                crate::haxe_string::haxe_string_from_bytes(hs_ptr, name.as_ptr(), name.len());
                // Push the HaxeString pointer into the array
                crate::haxe_array::haxe_array_push(
                    arr_ptr,
                    &hs_ptr as *const *mut HaxeString as *const u8,
                );
            }
        }

        arr_ptr as *mut u8
    }
}

/// Deep copy an anonymous object (creates independent clone)
#[no_mangle]
pub extern "C" fn rayzor_anon_copy(ptr: *mut u8) -> *mut u8 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let arc_ref = borrow_arc(ptr);
        let cloned_obj = (**arc_ref).clone();
        let arc = Arc::new(cloned_obj);
        let boxed = Box::new(arc);
        Box::into_raw(boxed) as *mut u8
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Box a raw u64 value as a DynamicValue pointer based on type_id
fn box_value_as_dynamic(type_id: u32, value: u64) -> *mut u8 {
    match TypeId(type_id) {
        t if t == TYPE_INT => crate::type_system::haxe_box_int_ptr(value as i64),
        t if t == TYPE_FLOAT => crate::type_system::haxe_box_float_ptr(f64::from_bits(value)),
        t if t == TYPE_BOOL => crate::type_system::haxe_box_bool_ptr(value != 0),
        t if t == TYPE_STRING => {
            crate::type_system::haxe_box_reference_ptr(value as *mut u8, TYPE_STRING.0)
        }
        t if t == TYPE_NULL => {
            let dv = DynamicValue {
                type_id: TYPE_NULL,
                value_ptr: std::ptr::null_mut(),
            };
            Box::into_raw(Box::new(dv)) as *mut u8
        }
        _ => {
            // Object or anon type — the value is a pointer
            crate::type_system::haxe_box_reference_ptr(value as *mut u8, type_id)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anon_new_and_drop() {
        let handle = rayzor_anon_new(DYNAMIC_SHAPE, 0);
        assert!(!handle.is_null());
        rayzor_anon_drop(handle);
    }

    #[test]
    fn test_inline_get_set() {
        // Create an inline object with 3 fields
        let handle = rayzor_anon_new(0, 3);
        rayzor_anon_set_field_by_index(handle, 0, 42);
        rayzor_anon_set_field_by_index(handle, 1, 100);
        rayzor_anon_set_field_by_index(handle, 2, f64::to_bits(2.78));

        assert_eq!(rayzor_anon_get_field_by_index(handle, 0), 42);
        assert_eq!(rayzor_anon_get_field_by_index(handle, 1), 100);
        assert_eq!(
            f64::from_bits(rayzor_anon_get_field_by_index(handle, 2)),
            2.78
        );

        rayzor_anon_drop(handle);
    }

    #[test]
    fn test_cow_clone() {
        // Create object with 2 fields
        let a = rayzor_anon_new(DYNAMIC_SHAPE, 0);
        rayzor_anon_drop(a);

        let a = rayzor_anon_new(0, 2);
        rayzor_anon_set_field_by_index(a, 0, 10);
        rayzor_anon_set_field_by_index(a, 1, 20);

        // Clone: a and b share backing
        let b = rayzor_anon_clone(a);

        // Mutate b — should COW (clone backing)
        rayzor_anon_set_field_by_index(b, 0, 99);

        // a unchanged, b modified
        assert_eq!(rayzor_anon_get_field_by_index(a, 0), 10);
        assert_eq!(rayzor_anon_get_field_by_index(b, 0), 99);

        rayzor_anon_drop(a);
        rayzor_anon_drop(b);
    }

    #[test]
    fn test_deep_copy() {
        let a = rayzor_anon_new(0, 2);
        rayzor_anon_set_field_by_index(a, 0, 42);
        rayzor_anon_set_field_by_index(a, 1, 99);

        let b = rayzor_anon_copy(a);
        rayzor_anon_set_field_by_index(b, 0, 1000);

        assert_eq!(rayzor_anon_get_field_by_index(a, 0), 42);
        assert_eq!(rayzor_anon_get_field_by_index(b, 0), 1000);

        rayzor_anon_drop(a);
        rayzor_anon_drop(b);
    }
}
