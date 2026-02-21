//! Runtime Type System for Dynamic values
//!
//! This module implements runtime type information (RTTI) for Haxe Dynamic values.
//! Each Dynamic value is represented as a tagged union: (type_id, value_ptr)
//!
//! ## Architecture
//!
//! - TypeId: Unique identifier for each type (Int, Float, Bool, String, classes, etc.)
//! - TypeInfo: Metadata for each type (size, alignment, toString, etc.)
//! - Type Registry: Global registry mapping TypeId -> TypeInfo
//!
//! ## Usage
//!
//! 1. Boxing: Convert a concrete value to Dynamic
//!    ```
//!    let dynamic = box_int(42);  // Returns (TYPE_INT, ptr)
//!    ```
//!
//! 2. Unboxing: Extract concrete value from Dynamic
//!    ```
//!    let value = unbox_int(dynamic);  // Returns 42
//!    ```
//!
//! 3. toString: Convert any Dynamic value to String
//!    ```
//!    let s = dynamic_to_string(dynamic);  // Dispatches based on type_id
//!    ```

use log::debug;
use std::collections::HashMap;
use std::sync::{Once, RwLock};

/// Ensures primitive types are registered exactly once
static INIT_PRIMITIVES: Once = Once::new();

/// Runtime type identifier
///
/// Each type in the Haxe type system gets a unique TypeId.
/// Primitive types have fixed IDs, classes get dynamic IDs.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub u32);

// Fixed type IDs for primitive types
pub const TYPE_VOID: TypeId = TypeId(0);
pub const TYPE_NULL: TypeId = TypeId(1);
pub const TYPE_BOOL: TypeId = TypeId(2);
pub const TYPE_INT: TypeId = TypeId(3);
pub const TYPE_FLOAT: TypeId = TypeId(4);
pub const TYPE_STRING: TypeId = TypeId(5);

// Starting ID for user-defined types (classes, enums, etc.)
pub const TYPE_USER_START: u32 = 1000;

/// Dynamic value: tagged union of (type_id, value_ptr)
///
/// This is the runtime representation of Haxe's Dynamic type.
/// The value_ptr points to heap-allocated memory containing the actual value.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DynamicValue {
    pub type_id: TypeId,
    pub value_ptr: *mut u8,
}

/// Function pointer type for toString implementations
///
/// Takes a pointer to the value and returns a String pointer (ptr + len)
pub type ToStringFn = unsafe extern "C" fn(*const u8) -> StringPtr;

/// String representation: pointer + length
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StringPtr {
    pub ptr: *const u8,
    pub len: usize,
}

/// Parameter type tag for RTTI
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ParamType {
    Int = 0,     // i64
    Float = 1,   // f64
    Bool = 2,    // bool
    String = 3,  // HaxeString pointer
    Object = 4,  // Generic pointer
    Dynamic = 5, // Unknown/generic type parameter — print as i64
}

/// Enum variant metadata
#[derive(Clone)]
pub struct EnumVariantInfo {
    /// Variant name (e.g., "Red", "Ok")
    pub name: &'static str,
    /// Number of parameters (0 for simple variants like Color.Red)
    pub param_count: usize,
    /// Parameter types for this variant (empty for parameterless variants)
    pub param_types: &'static [ParamType],
}

/// Enum type metadata
#[derive(Clone)]
pub struct EnumInfo {
    /// Enum type name (e.g., "Color", "Option")
    pub name: &'static str,
    /// Variant metadata indexed by discriminant
    pub variants: &'static [EnumVariantInfo],
}

/// Class field metadata
#[derive(Clone)]
pub struct ClassFieldInfo {
    /// Field name
    pub name: &'static str,
    /// Whether this is a static field
    pub is_static: bool,
}

/// Class type metadata
#[derive(Clone)]
pub struct ClassInfo {
    /// Qualified class name (e.g., "Main", "Animal")
    pub name: &'static str,
    /// Super class type id (None if no parent)
    pub super_type_id: Option<u32>,
    /// Instance fields (including inherited)
    pub instance_fields: &'static [&'static str],
    /// Static fields (own class only)
    pub static_fields: &'static [&'static str],
}

/// Type metadata
///
/// Contains all runtime information needed for a type:
/// - Size and alignment for memory allocation
/// - toString function for string conversion
/// - Type name for debugging
/// - Optional enum info for enum types
/// - Optional class info for class types
#[derive(Clone)]
pub struct TypeInfo {
    pub name: &'static str,
    pub size: usize,
    pub align: usize,
    pub to_string: ToStringFn,
    /// Enum-specific metadata (None for non-enum types)
    pub enum_info: Option<&'static EnumInfo>,
    /// Class-specific metadata (None for non-class types)
    pub class_info: Option<&'static ClassInfo>,
}

/// Global type registry
///
/// Maps TypeId -> TypeInfo for runtime type dispatch
pub(crate) static TYPE_REGISTRY: RwLock<Option<HashMap<TypeId, TypeInfo>>> = RwLock::new(None);

/// Global class name registry for Type.resolveClass
///
/// Maps qualified class name -> TypeId
static CLASS_NAME_REGISTRY: RwLock<Option<HashMap<String, u32>>> = RwLock::new(None);

/// Initialize the type registry with primitive types
pub fn init_type_system() {
    let mut registry = HashMap::new();

    // Register primitive types
    registry.insert(
        TYPE_VOID,
        TypeInfo {
            name: "Void",
            size: 0,
            align: 1,
            to_string: void_to_string,
            enum_info: None,
            class_info: None,
        },
    );

    registry.insert(
        TYPE_NULL,
        TypeInfo {
            name: "Null",
            size: 0,
            align: 1,
            to_string: null_to_string,
            enum_info: None,
            class_info: None,
        },
    );

    registry.insert(
        TYPE_BOOL,
        TypeInfo {
            name: "Bool",
            size: std::mem::size_of::<bool>(),
            align: std::mem::align_of::<bool>(),
            to_string: bool_to_string,
            enum_info: None,
            class_info: None,
        },
    );

    registry.insert(
        TYPE_INT,
        TypeInfo {
            name: "Int",
            size: std::mem::size_of::<i64>(),
            align: std::mem::align_of::<i64>(),
            to_string: int_to_string,
            enum_info: None,
            class_info: None,
        },
    );

    registry.insert(
        TYPE_FLOAT,
        TypeInfo {
            name: "Float",
            size: std::mem::size_of::<f64>(),
            align: std::mem::align_of::<f64>(),
            to_string: float_to_string,
            enum_info: None,
            class_info: None,
        },
    );

    registry.insert(
        TYPE_STRING,
        TypeInfo {
            name: "String",
            size: std::mem::size_of::<StringPtr>(),
            align: std::mem::align_of::<StringPtr>(),
            to_string: string_to_string,
            enum_info: None,
            class_info: None,
        },
    );

    *TYPE_REGISTRY.write().unwrap() = Some(registry);
}

/// Register a user-defined type (class, enum, etc.)
pub fn register_type(type_id: TypeId, info: TypeInfo) {
    // Ensure primitives are initialized first
    ensure_primitives_registered();

    let mut registry = TYPE_REGISTRY.write().unwrap();
    if let Some(ref mut map) = *registry {
        map.insert(type_id, info);
    }
}

/// Get type info for a TypeId
///
/// Primitive types (Void, Null, Bool, Int, Float, String) are lazily initialized
/// and always available. No explicit init_type_system() call is needed for primitives.
pub fn get_type_info(type_id: TypeId) -> Option<TypeInfo> {
    // Lazily initialize primitive types on first access
    ensure_primitives_registered();

    let registry = TYPE_REGISTRY.read().unwrap();
    registry.as_ref()?.get(&type_id).cloned()
}

/// Ensure primitive types are registered (called lazily)
fn ensure_primitives_registered() {
    INIT_PRIMITIVES.call_once(|| {
        init_type_system();
    });
}

/// Get enum variant name by type ID and discriminant
/// Returns None if not an enum type or discriminant is out of range
pub fn get_enum_variant_name(type_id: TypeId, discriminant: i64) -> Option<&'static str> {
    ensure_primitives_registered();
    let registry = TYPE_REGISTRY.read().unwrap();
    let type_info = registry.as_ref()?.get(&type_id)?;
    let enum_info = type_info.enum_info?;
    let idx = discriminant as usize;
    if idx < enum_info.variants.len() {
        Some(enum_info.variants[idx].name)
    } else {
        None
    }
}

/// Get enum variant info by type ID and discriminant
pub fn get_enum_variant_info(
    type_id: TypeId,
    discriminant: i64,
) -> Option<&'static EnumVariantInfo> {
    let registry = TYPE_REGISTRY.read().unwrap();
    let type_info = registry.as_ref()?.get(&type_id)?;
    let enum_info = type_info.enum_info?;
    let idx = discriminant as usize;
    enum_info.variants.get(idx)
}

/// Register an enum type with its variant metadata
#[no_mangle]
pub extern "C" fn haxe_register_enum(
    type_id: u32,
    name_ptr: *const u8,
    name_len: usize,
    variants_ptr: *const EnumVariantInfo,
    variants_len: usize,
) {
    // Safety: We trust the compiler to pass valid pointers
    // The variant data must be static (lifetime 'static)
    unsafe {
        let name_slice = std::slice::from_raw_parts(name_ptr, name_len);
        let name_str = std::str::from_utf8_unchecked(name_slice);
        // SAFETY: The compiler ensures this data lives for 'static
        let name: &'static str = std::mem::transmute(name_str);

        let variants: &'static [EnumVariantInfo] =
            std::slice::from_raw_parts(variants_ptr, variants_len);

        // Create a static EnumInfo - we need to leak it since TypeInfo expects &'static
        let enum_info = Box::leak(Box::new(EnumInfo { name, variants }));

        let type_info = TypeInfo {
            name,
            size: std::mem::size_of::<i64>(), // Enums are represented as i64 discriminants
            align: std::mem::align_of::<i64>(),
            to_string: enum_to_string,
            enum_info: Some(enum_info),
            class_info: None,
        };

        register_type(TypeId(type_id), type_info);
    }
}

// ============================================================================
// Rust-native enum RTTI registration (called from compiler backends directly)
// ============================================================================

/// Register enum RTTI directly from MIR metadata, bypassing generated code.
/// `variants` is a slice of (name, param_count, param_types) tuples.
pub fn register_enum_from_mir(
    type_id: u32,
    name: &str,
    variants: &[(String, usize, Vec<ParamType>)],
) {
    let enum_name_static: &'static str = Box::leak(name.to_string().into_boxed_str());

    let variant_infos: Vec<EnumVariantInfo> = variants
        .iter()
        .map(|(vname, param_count, param_types)| EnumVariantInfo {
            name: Box::leak(vname.clone().into_boxed_str()),
            param_count: *param_count,
            param_types: Box::leak(param_types.clone().into_boxed_slice()),
        })
        .collect();

    let variants_static: &'static [EnumVariantInfo] = Box::leak(variant_infos.into_boxed_slice());

    let enum_info = Box::leak(Box::new(EnumInfo {
        name: enum_name_static,
        variants: variants_static,
    }));

    let type_info = TypeInfo {
        name: enum_name_static,
        size: std::mem::size_of::<i64>(),
        align: std::mem::align_of::<i64>(),
        to_string: enum_to_string,
        enum_info: Some(enum_info),
        class_info: None,
    };

    register_type(TypeId(type_id), type_info);
}

// ============================================================================
// Class RTTI registration (called from compiler backends)
// ============================================================================

/// Register class RTTI directly from MIR metadata.
/// `instance_fields` are all instance field names (including inherited).
/// `static_fields` are own static field names.
pub fn register_class_from_mir(
    type_id: u32,
    name: &str,
    super_type_id: Option<u32>,
    instance_fields: &[String],
    static_fields: &[String],
) {
    let class_name_static: &'static str = Box::leak(name.to_string().into_boxed_str());

    let instance_fields_static: &'static [&'static str] = Box::leak(
        instance_fields
            .iter()
            .map(|f| -> &'static str { Box::leak(f.clone().into_boxed_str()) })
            .collect::<Vec<&'static str>>()
            .into_boxed_slice(),
    );

    let static_fields_static: &'static [&'static str] = Box::leak(
        static_fields
            .iter()
            .map(|f| -> &'static str { Box::leak(f.clone().into_boxed_str()) })
            .collect::<Vec<&'static str>>()
            .into_boxed_slice(),
    );

    let class_info = Box::leak(Box::new(ClassInfo {
        name: class_name_static,
        super_type_id,
        instance_fields: instance_fields_static,
        static_fields: static_fields_static,
    }));

    let type_info = TypeInfo {
        name: class_name_static,
        size: 0, // Struct size is not tracked here (computed by backend)
        align: 8,
        to_string: void_to_string,
        enum_info: None,
        class_info: Some(class_info),
    };

    register_type(TypeId(type_id), type_info);

    // Also register in the name-to-TypeId reverse map
    let mut guard = CLASS_NAME_REGISTRY.write().unwrap();
    let registry = guard.get_or_insert_with(HashMap::new);
    registry.insert(name.to_string(), type_id);
}

// ============================================================================
// Class RTTI query functions (called from JIT'd code)
// ============================================================================

/// Helper: allocate a HaxeString from a &str, using the C API.
unsafe fn alloc_haxe_string(s: &str) -> *mut u8 {
    let hs_layout = std::alloc::Layout::new::<crate::haxe_string::HaxeString>();
    let hs_ptr = std::alloc::alloc(hs_layout) as *mut crate::haxe_string::HaxeString;
    if hs_ptr.is_null() {
        return std::ptr::null_mut();
    }
    crate::haxe_string::haxe_string_from_bytes(hs_ptr, s.as_ptr(), s.len());
    hs_ptr as *mut u8
}

/// Helper: build a HaxeArray of HaxeStrings from a slice of &str.
unsafe fn build_string_array(names: &[&str]) -> *mut u8 {
    let arr_layout = std::alloc::Layout::new::<crate::haxe_array::HaxeArray>();
    let arr_ptr = std::alloc::alloc(arr_layout) as *mut crate::haxe_array::HaxeArray;
    if arr_ptr.is_null() {
        return std::ptr::null_mut();
    }
    crate::haxe_array::haxe_array_new(
        arr_ptr,
        std::mem::size_of::<*mut crate::haxe_string::HaxeString>(),
    );
    for name in names {
        let hs_ptr = alloc_haxe_string(name);
        if !hs_ptr.is_null() {
            crate::haxe_array::haxe_array_push(arr_ptr, &hs_ptr as *const *mut u8 as *const u8);
        }
    }
    arr_ptr as *mut u8
}

/// Type.getClassName(c) -> String
/// Takes a type_id (i64), returns the class name as a HaxeString pointer.
#[no_mangle]
pub extern "C" fn haxe_type_get_class_name(type_id: i64) -> *mut u8 {
    let guard = TYPE_REGISTRY.read().unwrap();
    if let Some(registry) = guard.as_ref() {
        if let Some(type_info) = registry.get(&TypeId(type_id as u32)) {
            if let Some(class_info) = &type_info.class_info {
                return unsafe { alloc_haxe_string(class_info.name) };
            }
        }
    }
    std::ptr::null_mut()
}

/// Type.getSuperClass(c) -> Class<Dynamic> (returns super's type_id, or -1 if none)
#[no_mangle]
pub extern "C" fn haxe_type_get_super_class(type_id: i64) -> i64 {
    let guard = TYPE_REGISTRY.read().unwrap();
    if let Some(registry) = guard.as_ref() {
        if let Some(type_info) = registry.get(&TypeId(type_id as u32)) {
            if let Some(class_info) = &type_info.class_info {
                if let Some(super_id) = class_info.super_type_id {
                    return super_id as i64;
                }
            }
        }
    }
    -1
}

/// Type.getInstanceFields(c) -> Array<String>
/// Returns an array of instance field names (including inherited).
#[no_mangle]
pub extern "C" fn haxe_type_get_instance_fields(type_id: i64) -> *mut u8 {
    let guard = TYPE_REGISTRY.read().unwrap();
    if let Some(registry) = guard.as_ref() {
        if let Some(type_info) = registry.get(&TypeId(type_id as u32)) {
            if let Some(class_info) = &type_info.class_info {
                return unsafe { build_string_array(class_info.instance_fields) };
            }
        }
    }
    unsafe { build_string_array(&[]) }
}

/// Type.getClassFields(c) -> Array<String>
/// Returns an array of static field names (own class only).
#[no_mangle]
pub extern "C" fn haxe_type_get_class_fields(type_id: i64) -> *mut u8 {
    let guard = TYPE_REGISTRY.read().unwrap();
    if let Some(registry) = guard.as_ref() {
        if let Some(type_info) = registry.get(&TypeId(type_id as u32)) {
            if let Some(class_info) = &type_info.class_info {
                return unsafe { build_string_array(class_info.static_fields) };
            }
        }
    }
    unsafe { build_string_array(&[]) }
}

/// Type.resolveClass(name) -> Class<Dynamic> (returns type_id, or -1 if not found)
#[no_mangle]
pub extern "C" fn haxe_type_resolve_class(name_ptr: *mut u8) -> i64 {
    if name_ptr.is_null() {
        return -1;
    }
    let name = unsafe {
        let hs = &*(name_ptr as *const crate::haxe_string::HaxeString);
        if hs.ptr.is_null() || hs.len == 0 {
            return -1;
        }
        let bytes = std::slice::from_raw_parts(hs.ptr, hs.len);
        String::from_utf8_lossy(bytes).to_string()
    };
    let guard = CLASS_NAME_REGISTRY.read().unwrap();
    if let Some(registry) = guard.as_ref() {
        if let Some(&type_id) = registry.get(&name) {
            return type_id as i64;
        }
    }
    -1
}

/// Type.getEnumConstructs(e) -> Array<String>
/// Takes an enum type_id, returns array of constructor names.
#[no_mangle]
pub extern "C" fn haxe_type_get_enum_constructs(type_id: i64) -> *mut u8 {
    let guard = TYPE_REGISTRY.read().unwrap();
    if let Some(registry) = guard.as_ref() {
        if let Some(type_info) = registry.get(&TypeId(type_id as u32)) {
            if let Some(enum_info) = &type_info.enum_info {
                let names: Vec<&str> = enum_info.variants.iter().map(|v| v.name).collect();
                return unsafe { build_string_array(&names) };
            }
        }
    }
    unsafe { build_string_array(&[]) }
}

/// Type.getEnumName(e) -> String
#[no_mangle]
pub extern "C" fn haxe_type_get_enum_name(type_id: i64) -> *mut u8 {
    let guard = TYPE_REGISTRY.read().unwrap();
    if let Some(registry) = guard.as_ref() {
        if let Some(type_info) = registry.get(&TypeId(type_id as u32)) {
            if let Some(enum_info) = &type_info.enum_info {
                return unsafe { alloc_haxe_string(enum_info.name) };
            }
        }
    }
    std::ptr::null_mut()
}

/// Type.createEnum(e, constr, ?params) -> T
/// Creates an enum value dynamically by constructor name.
/// For parameterless constructors, returns the tag as i64 (unboxed).
/// For constructors with parameters, allocates boxed enum [tag:i32][pad:i32][fields...].
#[no_mangle]
pub extern "C" fn haxe_type_create_enum(
    type_id: i64,
    constr_ptr: *mut u8,
    params_ptr: *mut u8,
) -> i64 {
    let constr_name = if constr_ptr.is_null() {
        return 0;
    } else {
        unsafe {
            let hs = &*(constr_ptr as *const crate::haxe_string::HaxeString);
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(hs.ptr, hs.len))
        }
    };

    let guard = TYPE_REGISTRY.read().unwrap();
    if let Some(registry) = guard.as_ref() {
        if let Some(type_info) = registry.get(&TypeId(type_id as u32)) {
            if let Some(enum_info) = &type_info.enum_info {
                for (idx, variant) in enum_info.variants.iter().enumerate() {
                    if variant.name == constr_name {
                        return create_enum_value(idx as i32, variant.param_count, params_ptr);
                    }
                }
            }
        }
    }
    0
}

/// Type.createEnumIndex(e, index, ?params) -> T
/// Creates an enum value dynamically by constructor index.
#[no_mangle]
pub extern "C" fn haxe_type_create_enum_index(
    type_id: i64,
    index: i64,
    params_ptr: *mut u8,
) -> i64 {
    let guard = TYPE_REGISTRY.read().unwrap();
    if let Some(registry) = guard.as_ref() {
        if let Some(type_info) = registry.get(&TypeId(type_id as u32)) {
            if let Some(enum_info) = &type_info.enum_info {
                if let Some(variant) = enum_info.variants.get(index as usize) {
                    return create_enum_value(index as i32, variant.param_count, params_ptr);
                }
            }
        }
    }
    0
}

/// Helper: create an enum value (unboxed tag or boxed struct)
fn create_enum_value(tag: i32, param_count: usize, params_ptr: *mut u8) -> i64 {
    if param_count == 0 {
        // Unboxed: just the tag
        return tag as i64;
    }

    // Boxed: allocate [tag:i32][pad:i32][field0:i64][field1:i64]...
    let size = 8 + param_count * 8;
    unsafe {
        let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            return 0;
        }
        // Write tag
        *(ptr as *mut i32) = tag;

        // Copy parameters from the HaxeArray if provided
        if !params_ptr.is_null() {
            let arr = &*(params_ptr as *const crate::haxe_array::HaxeArray);
            for i in 0..param_count.min(arr.len) {
                let val = crate::haxe_array::haxe_array_get_i64(
                    params_ptr as *const crate::haxe_array::HaxeArray,
                    i,
                );
                let field_ptr = ptr.add(8 + i * 8);
                *(field_ptr as *mut i64) = val;
            }
        }

        ptr as i64
    }
}

// ============================================================================
// Simpler per-variant registration API (easier to call from generated code)
// ============================================================================

/// Variant data collected during registration: (name, param_count, param_types)
type EnumVariantBuilder = (String, usize, Vec<ParamType>);

/// Enum registration state: (enum_name, variants, expected_variant_count)
type EnumBuilderEntry = (String, Vec<EnumVariantBuilder>, usize);

/// Storage for enum registrations in progress
/// Maps type_id -> EnumBuilderEntry
static ENUM_BUILDER: RwLock<Option<HashMap<u32, EnumBuilderEntry>>> = RwLock::new(None);

/// Start registering an enum type - call this first, then call register_enum_variant for each variant
/// Finally call register_enum_finish to complete registration
/// Note: name_str is a HaxeString pointer (from IrValue::String), not raw *const u8
#[no_mangle]
pub extern "C" fn haxe_register_enum_start(
    type_id: u32,
    name_str: *const crate::string::HaxeString,
    _name_len: usize, // Kept for ABI compatibility, but we use HaxeString.len
    variant_count: usize,
) {
    unsafe {
        // Extract the actual string data from HaxeString
        let name = if name_str.is_null() {
            String::from("<null>")
        } else {
            let haxe_str = &*name_str;
            let name_slice = std::slice::from_raw_parts(haxe_str.ptr as *const u8, haxe_str.len);
            String::from_utf8_lossy(name_slice).to_string()
        };

        let mut builder = ENUM_BUILDER.write().unwrap();
        if builder.is_none() {
            *builder = Some(HashMap::new());
        }
        builder.as_mut().unwrap().insert(
            type_id,
            (name, Vec::with_capacity(variant_count), variant_count),
        );
    }
}

/// Register a single enum variant - call after register_enum_start for each variant
/// Note: name_str is a HaxeString pointer (from IrValue::String), not raw *const u8
/// param_types_ptr: pointer to array of ParamType (u8), or null if no params
#[no_mangle]
pub extern "C" fn haxe_register_enum_variant(
    type_id: u32,
    _variant_index: usize,
    name_str: *const crate::string::HaxeString,
    _name_len: usize, // Kept for ABI compatibility, but we use HaxeString.len
    param_count: usize,
    param_types_ptr: *const u8, // Array of ParamType (u8) values
) {
    unsafe {
        // Extract the actual string data from HaxeString
        let name = if name_str.is_null() {
            String::from("<null>")
        } else {
            let haxe_str = &*name_str;
            let name_slice = std::slice::from_raw_parts(haxe_str.ptr as *const u8, haxe_str.len);
            String::from_utf8_lossy(name_slice).to_string()
        };

        // Extract param types
        let param_types: Vec<ParamType> = if param_types_ptr.is_null() || param_count == 0 {
            Vec::new()
        } else {
            let types_slice = std::slice::from_raw_parts(param_types_ptr, param_count);
            types_slice
                .iter()
                .map(|&t| match t {
                    0 => ParamType::Int,
                    1 => ParamType::Float,
                    2 => ParamType::Bool,
                    3 => ParamType::String,
                    5 => ParamType::Dynamic,
                    _ => ParamType::Object,
                })
                .collect()
        };

        let mut builder = ENUM_BUILDER.write().unwrap();
        if let Some(ref mut map) = *builder {
            if let Some((_, variants, _)) = map.get_mut(&type_id) {
                variants.push((name, param_count, param_types));
            }
        }
    }
}

/// Finish enum registration - call after all variants have been registered
/// This creates the final TypeInfo and registers it
#[no_mangle]
pub extern "C" fn haxe_register_enum_finish(type_id: u32) {
    let mut builder = ENUM_BUILDER.write().unwrap();
    if let Some(ref mut map) = *builder {
        if let Some((enum_name, variants, _)) = map.remove(&type_id) {
            // Convert to static storage
            let enum_name_static: &'static str = Box::leak(enum_name.into_boxed_str());

            // Build static variant info array
            let variant_infos: Vec<EnumVariantInfo> = variants
                .into_iter()
                .map(|(name, param_count, param_types)| EnumVariantInfo {
                    name: Box::leak(name.into_boxed_str()),
                    param_count,
                    param_types: Box::leak(param_types.into_boxed_slice()),
                })
                .collect();

            let variants_static: &'static [EnumVariantInfo] =
                Box::leak(variant_infos.into_boxed_slice());

            let enum_info = Box::leak(Box::new(EnumInfo {
                name: enum_name_static,
                variants: variants_static,
            }));

            let type_info = TypeInfo {
                name: enum_name_static,
                size: std::mem::size_of::<i64>(),
                align: std::mem::align_of::<i64>(),
                to_string: enum_to_string,
                enum_info: Some(enum_info),
                class_info: None,
            };

            register_type(TypeId(type_id), type_info);

            debug!(
                "Registered enum '{}' with {} variants at type_id {}",
                enum_name_static,
                variants_static.len(),
                type_id
            );
        }
    }
}

/// toString implementation for enum types
/// Takes a pointer to (type_id: u32, discriminant: i64) tuple
unsafe extern "C" fn enum_to_string(value_ptr: *const u8) -> StringPtr {
    // Enum values are stored as just the discriminant (i64)
    // We need to look up the type from context - for now return the discriminant as string
    let discriminant = *(value_ptr as *const i64);

    // For now, just format the discriminant - proper lookup requires type_id context
    // This will be improved when we have proper Dynamic boxing with type_id
    let s = format!("{}", discriminant);
    let leaked = Box::leak(s.into_boxed_str());
    StringPtr {
        ptr: leaked.as_ptr(),
        len: leaked.len(),
    }
}

/// Get enum variant name as a HaxeString pointer
/// Get the discriminant index of an enum value.
/// For unboxed enums (is_boxed=0): value IS the discriminant, return directly.
/// For boxed enums (is_boxed!=0): value is a pointer, read i32 tag from offset 0.
#[no_mangle]
pub extern "C" fn haxe_enum_get_index(value: i64, is_boxed: i32) -> i64 {
    if is_boxed == 0 {
        value
    } else {
        let ptr = value as *const u8;
        if ptr.is_null() {
            return -1;
        }
        unsafe { *(ptr as *const i32) as i64 }
    }
}

/// Get the name of an enum value.
/// For unboxed enums (is_boxed=0): value is the discriminant.
/// For boxed enums (is_boxed!=0): value is a pointer, read i32 tag from offset 0.
#[no_mangle]
pub extern "C" fn haxe_enum_get_name(
    type_id: u32,
    value: i64,
    is_boxed: i32,
) -> *mut crate::haxe_string::HaxeString {
    let discriminant = if is_boxed == 0 {
        value
    } else {
        let ptr = value as *const u8;
        if ptr.is_null() {
            return std::ptr::null_mut();
        }
        unsafe { *(ptr as *const i32) as i64 }
    };
    haxe_enum_variant_name(type_id, discriminant)
}

/// Returns the variant name for the given type_id and discriminant
/// Returns null if not an enum or discriminant is out of range
#[no_mangle]
pub extern "C" fn haxe_enum_variant_name(
    type_id: u32,
    discriminant: i64,
) -> *mut crate::haxe_string::HaxeString {
    use crate::haxe_string::HaxeString;

    if let Some(name) = get_enum_variant_name(TypeId(type_id), discriminant) {
        // Create a HaxeString from the static variant name
        // cap=0 indicates static/borrowed string that shouldn't be freed
        let result = Box::new(HaxeString {
            ptr: name.as_ptr() as *mut u8,
            len: name.len(),
            cap: 0, // Static string, don't free
        });
        Box::into_raw(result)
    } else {
        std::ptr::null_mut()
    }
}

/// Get name of a boxed enum value (heap-allocated with tag at offset 0).
/// Reads the i32 tag from the struct, then looks up the variant name via RTTI.
#[no_mangle]
pub extern "C" fn haxe_enum_get_name_boxed(
    type_id: u32,
    ptr: *const u8,
) -> *mut crate::haxe_string::HaxeString {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let tag = *(ptr as *const i32) as i64;
        haxe_enum_variant_name(type_id, tag)
    }
}

/// Get parameters of an enum value as a HaxeArray.
/// For unboxed enums (is_boxed=0): returns empty array.
/// For boxed enums (is_boxed!=0): reads fields from heap struct and boxes them as Dynamic.
#[no_mangle]
pub extern "C" fn haxe_enum_get_parameters(
    type_id: u32,
    value: i64,
    is_boxed: i32,
) -> *mut crate::haxe_array::HaxeArray {
    use crate::haxe_array::HaxeArray;
    use std::alloc::{alloc, Layout};

    // Allocate a HaxeArray on the heap
    let arr = unsafe {
        let layout = Layout::new::<HaxeArray>();
        let ptr = alloc(layout) as *mut HaxeArray;
        if ptr.is_null() {
            panic!("Failed to allocate HaxeArray for getParameters");
        }
        crate::haxe_array::haxe_array_new(ptr, 8); // elem_size=8 for i64/pointer elements
        ptr
    };

    if is_boxed == 0 {
        // Unboxed enum: no parameters, return empty array
        return arr;
    }

    // Boxed enum: value is a pointer to [tag:i32][pad:i32][field0:i64][field1:i64]...
    let ptr = value as *const u8;
    if ptr.is_null() {
        return arr;
    }

    unsafe {
        let tag = *(ptr as *const i32) as i64;
        if let Some(variant_info) = get_enum_variant_info(TypeId(type_id), tag) {
            for i in 0..variant_info.param_count {
                let field_ptr = ptr.add(8 + i * 8);
                let raw_val = *(field_ptr as *const i64);
                let param_type = variant_info
                    .param_types
                    .get(i)
                    .copied()
                    .unwrap_or(ParamType::Dynamic);
                // Box the field value as Dynamic based on its type
                let boxed = match param_type {
                    ParamType::Int => haxe_box_int_ptr(raw_val),
                    ParamType::Float => haxe_box_float_ptr(f64::from_bits(raw_val as u64)),
                    ParamType::Bool => haxe_box_bool_ptr(raw_val != 0),
                    // String and Object fields are already pointers, pass through
                    ParamType::String | ParamType::Object | ParamType::Dynamic => {
                        haxe_box_int_ptr(raw_val)
                    }
                };
                crate::haxe_array::haxe_array_push_i64(arr, boxed as i64);
            }
        }
    }

    arr
}

/// Walk the class hierarchy to check if actual_type_id is or extends expected_type_id.
fn type_id_matches_with_hierarchy(actual_type_id: i64, expected_type_id: i64) -> bool {
    if actual_type_id == expected_type_id {
        return true;
    }
    let registry = TYPE_REGISTRY.read().unwrap();
    if let Some(ref registry) = *registry {
        let mut current = TypeId(actual_type_id as u32);
        while let Some(type_info) = registry.get(&current) {
            if let Some(class_info) = &type_info.class_info {
                if let Some(parent_id) = class_info.super_type_id {
                    if parent_id as i64 == expected_type_id {
                        return true;
                    }
                    current = TypeId(parent_id);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
    false
}

/// Runtime type check for Dynamic/boxed values.
/// Checks if a boxed DynamicValue has the given type_id, walking the class hierarchy.
/// Used by `Std.is()` and `(expr is Type)` for Dynamic-typed values.
#[no_mangle]
pub extern "C" fn haxe_std_is(value_ptr: *mut u8, expected_type_id: i64) -> bool {
    if value_ptr.is_null() {
        return false;
    }
    let actual_type_id = unsafe {
        let dynamic = *(value_ptr as *const DynamicValue);
        dynamic.type_id.0 as i64
    };
    type_id_matches_with_hierarchy(actual_type_id, expected_type_id)
}

/// Runtime downcast for Dynamic/boxed values.
/// Returns the value pointer if the type matches (with hierarchy walking), null otherwise.
/// Used by `Std.downcast()`.
#[no_mangle]
pub extern "C" fn haxe_std_downcast(value_ptr: *mut u8, expected_type_id: i64) -> *mut u8 {
    if value_ptr.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let dynamic = *(value_ptr as *const DynamicValue);
        if type_id_matches_with_hierarchy(dynamic.type_id.0 as i64, expected_type_id) {
            dynamic.value_ptr
        } else {
            std::ptr::null_mut()
        }
    }
}

// ============================================================================
// Type API enum wrappers (accept boxed DynamicValue* from Haxe Type API)
// ============================================================================

/// Type.enumIndex(e:EnumValue):Int — get enum index from raw i64 value
/// For simple enums: value is the discriminant (= index)
/// For boxed enums: value is a pointer to [tag:i32][pad:i32][fields...]
#[no_mangle]
pub extern "C" fn haxe_type_enum_index(value: i64) -> i64 {
    // Simple (unboxed) enums: discriminant IS the index
    haxe_enum_get_index(value, 0)
}

/// Type.enumConstructor(e:EnumValue):String — get constructor name
/// Takes raw i64 value + type_id (injected by compiler)
#[no_mangle]
pub extern "C" fn haxe_type_enum_constructor(
    value: i64,
    type_id: i32,
) -> *mut crate::haxe_string::HaxeString {
    haxe_enum_get_name(type_id as u32, value, 0)
}

/// Type.enumParameters(e:EnumValue):Array<Dynamic> — get parameters
/// Takes raw i64 value + type_id (injected by compiler)
/// For simple enums, always returns empty array
#[no_mangle]
pub extern "C" fn haxe_type_enum_parameters(
    value: i64,
    type_id: i32,
) -> *mut crate::haxe_array::HaxeArray {
    haxe_enum_get_parameters(type_id as u32, value, 0)
}

/// Trace an enum value by type_id and discriminant
/// Prints the variant name if available, otherwise the discriminant
#[no_mangle]
pub extern "C" fn haxe_trace_enum(type_id: i64, discriminant: i64) {
    if let Some(name) = get_enum_variant_name(TypeId(type_id as u32), discriminant) {
        println!("{}", name);
    } else {
        // Fallback to discriminant if enum not registered
        println!("{}", discriminant);
    }
}

/// Trace a boxed enum value (heap-allocated with parameters)
/// Memory layout: [tag:i32][pad:i32][field0:i64][field1:i64]...
/// Prints "VariantName(param1, param2, ...)" format
#[no_mangle]
pub extern "C" fn haxe_trace_enum_boxed(type_id: u32, ptr: *const u8) {
    if ptr.is_null() {
        println!("null");
        return;
    }

    unsafe {
        // Read tag (discriminant) from offset 0
        let tag = *(ptr as *const i32);

        // Look up variant info from RTTI
        let variant_info = match get_enum_variant_info(TypeId(type_id), tag as i64) {
            Some(info) => info,
            None => {
                // Fallback if enum not registered
                println!("<enum {}::{}>", type_id, tag);
                return;
            }
        };

        let variant_name = variant_info.name;
        let param_count = variant_info.param_count;
        let param_types = variant_info.param_types;

        if param_count == 0 {
            println!("{}", variant_name);
        } else {
            print!("{}(", variant_name);
            for i in 0..param_count {
                if i > 0 {
                    print!(", ");
                }
                // Read field at offset 8 + i * 8
                let field_ptr = ptr.add(8 + i * 8);

                // Get param type (default to Int if not available)
                let param_type = param_types.get(i).copied().unwrap_or(ParamType::Int);

                match param_type {
                    ParamType::Int => {
                        let val = *(field_ptr as *const i64);
                        print!("{}", val);
                    }
                    ParamType::Float => {
                        let val = *(field_ptr as *const f64);
                        print!("{}", val);
                    }
                    ParamType::Bool => {
                        let val = *(field_ptr as *const i64) != 0;
                        print!("{}", val);
                    }
                    ParamType::String => {
                        // Field is a pointer to HaxeString
                        let str_ptr = *(field_ptr as *const *const crate::haxe_string::HaxeString);
                        if str_ptr.is_null() {
                            print!("null");
                        } else {
                            let haxe_str = &*str_ptr;
                            let bytes =
                                std::slice::from_raw_parts(haxe_str.ptr as *const u8, haxe_str.len);
                            match std::str::from_utf8(bytes) {
                                Ok(s) => print!("\"{}\"", s),
                                Err(_) => print!("<invalid utf8>"),
                            }
                        }
                    }
                    ParamType::Object => {
                        let val = *(field_ptr as *const i64);
                        print!("<object@0x{:x}>", val);
                    }
                    ParamType::Dynamic => {
                        // Generic type parameter — print raw i64 value
                        let val = *(field_ptr as *const i64);
                        print!("{}", val);
                    }
                }
            }
            println!(")");
        }
    }
}

/// Trace a boxed enum value with explicit param types from type inference.
/// Called when the compiler knows the concrete types at the call site (e.g. Result<Int, String>).
/// param_types_ptr: pointer to array of ParamType (u8) values, one per variant field.
/// param_count: number of entries in param_types_ptr.
#[no_mangle]
pub extern "C" fn haxe_trace_enum_boxed_typed(
    type_id: u32,
    ptr: *const u8,
    param_types_ptr: *const u8,
    param_count: usize,
) {
    if ptr.is_null() {
        println!("null");
        return;
    }

    unsafe {
        let tag = *(ptr as *const i32);

        // Look up variant name from RTTI (we still need the name)
        let variant_name = get_enum_variant_info(TypeId(type_id), tag as i64)
            .map(|info| info.name.to_string())
            .unwrap_or_else(|| format!("variant{}", tag));

        // Build param types from the caller-provided array
        let caller_types: Vec<ParamType> = if param_types_ptr.is_null() || param_count == 0 {
            Vec::new()
        } else {
            let types_slice = std::slice::from_raw_parts(param_types_ptr, param_count);
            types_slice
                .iter()
                .map(|&t| match t {
                    0 => ParamType::Int,
                    1 => ParamType::Float,
                    2 => ParamType::Bool,
                    3 => ParamType::String,
                    5 => ParamType::Dynamic,
                    _ => ParamType::Object,
                })
                .collect()
        };

        if caller_types.is_empty() {
            println!("{}", variant_name);
        } else {
            print!("{}(", variant_name);
            for (i, &param_type) in caller_types.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                let field_ptr = ptr.add(8 + i * 8);

                match param_type {
                    ParamType::Int => {
                        let val = *(field_ptr as *const i64);
                        print!("{}", val);
                    }
                    ParamType::Float => {
                        let val = *(field_ptr as *const f64);
                        print!("{}", val);
                    }
                    ParamType::Bool => {
                        let val = *(field_ptr as *const i64) != 0;
                        print!("{}", val);
                    }
                    ParamType::String => {
                        let str_ptr = *(field_ptr as *const *const crate::haxe_string::HaxeString);
                        if str_ptr.is_null() {
                            print!("null");
                        } else {
                            let haxe_str = &*str_ptr;
                            let bytes =
                                std::slice::from_raw_parts(haxe_str.ptr as *const u8, haxe_str.len);
                            match std::str::from_utf8(bytes) {
                                Ok(s) => print!("\"{}\"", s),
                                Err(_) => print!("<invalid utf8>"),
                            }
                        }
                    }
                    ParamType::Object | ParamType::Dynamic => {
                        let val = *(field_ptr as *const i64);
                        print!("{}", val);
                    }
                }
            }
            println!(")");
        }
    }
}

// ============================================================================
// toString implementations for primitive types
// ============================================================================

unsafe extern "C" fn void_to_string(_value_ptr: *const u8) -> StringPtr {
    let s = "void";
    StringPtr {
        ptr: s.as_ptr(),
        len: s.len(),
    }
}

unsafe extern "C" fn null_to_string(_value_ptr: *const u8) -> StringPtr {
    let s = "null";
    StringPtr {
        ptr: s.as_ptr(),
        len: s.len(),
    }
}

unsafe extern "C" fn bool_to_string(value_ptr: *const u8) -> StringPtr {
    let value = *(value_ptr as *const bool);
    let s = if value { "true" } else { "false" };
    StringPtr {
        ptr: s.as_ptr(),
        len: s.len(),
    }
}

unsafe extern "C" fn int_to_string(value_ptr: *const u8) -> StringPtr {
    let value = *(value_ptr as *const i64);
    let s = value.to_string();
    // UNSAFE: Leaking memory! Need proper string management
    // TODO: Use a string pool or return owned strings
    let s_static = Box::leak(s.into_boxed_str());
    StringPtr {
        ptr: s_static.as_ptr(),
        len: s_static.len(),
    }
}

unsafe extern "C" fn float_to_string(value_ptr: *const u8) -> StringPtr {
    let value = *(value_ptr as *const f64);
    let s = value.to_string();
    // UNSAFE: Leaking memory! Need proper string management
    let s_static = Box::leak(s.into_boxed_str());
    StringPtr {
        ptr: s_static.as_ptr(),
        len: s_static.len(),
    }
}

unsafe extern "C" fn string_to_string(value_ptr: *const u8) -> StringPtr {
    // String is already a StringPtr, just return it
    *(value_ptr as *const StringPtr)
}

// ============================================================================
// Boxing functions: Convert concrete values to Dynamic
// ============================================================================

/// Box an Int as Dynamic
#[no_mangle]
pub extern "C" fn haxe_box_int(value: i64) -> DynamicValue {
    unsafe {
        let ptr = libc::malloc(std::mem::size_of::<i64>()) as *mut i64;
        *ptr = value;
        DynamicValue {
            type_id: TYPE_INT,
            value_ptr: ptr as *mut u8,
        }
    }
}

/// Box a Float as Dynamic
#[no_mangle]
pub extern "C" fn haxe_box_float(value: f64) -> DynamicValue {
    unsafe {
        let ptr = libc::malloc(std::mem::size_of::<f64>()) as *mut f64;
        *ptr = value;
        DynamicValue {
            type_id: TYPE_FLOAT,
            value_ptr: ptr as *mut u8,
        }
    }
}

/// Box a Bool as Dynamic
#[no_mangle]
pub extern "C" fn haxe_box_bool(value: bool) -> DynamicValue {
    unsafe {
        let ptr = libc::malloc(std::mem::size_of::<bool>()) as *mut bool;
        *ptr = value;
        DynamicValue {
            type_id: TYPE_BOOL,
            value_ptr: ptr as *mut u8,
        }
    }
}

/// Box a String as Dynamic
#[no_mangle]
pub extern "C" fn haxe_box_string(str_ptr: *const u8, len: usize) -> DynamicValue {
    unsafe {
        let ptr = libc::malloc(std::mem::size_of::<StringPtr>()) as *mut StringPtr;
        *ptr = StringPtr { ptr: str_ptr, len };
        DynamicValue {
            type_id: TYPE_STRING,
            value_ptr: ptr as *mut u8,
        }
    }
}

/// Box null as Dynamic
#[no_mangle]
pub extern "C" fn haxe_box_null() -> DynamicValue {
    DynamicValue {
        type_id: TYPE_NULL,
        value_ptr: std::ptr::null_mut(),
    }
}

// ============================================================================
// Unboxing functions: Extract concrete values from Dynamic
// ============================================================================

/// Unbox a Dynamic as Int (handles Float→Int truncation, returns 0 for other types)
#[no_mangle]
pub extern "C" fn haxe_unbox_int(dynamic: DynamicValue) -> i64 {
    if dynamic.type_id == TYPE_INT {
        unsafe { *(dynamic.value_ptr as *const i64) }
    } else if dynamic.type_id == TYPE_FLOAT {
        unsafe { *(dynamic.value_ptr as *const f64) as i64 }
    } else if dynamic.type_id == TYPE_BOOL {
        unsafe {
            if *(dynamic.value_ptr as *const bool) {
                1
            } else {
                0
            }
        }
    } else {
        0
    }
}

/// Unbox a Dynamic as Float (handles Int→Float promotion, returns 0.0 for other types)
#[no_mangle]
pub extern "C" fn haxe_unbox_float(dynamic: DynamicValue) -> f64 {
    if dynamic.type_id == TYPE_FLOAT {
        unsafe { *(dynamic.value_ptr as *const f64) }
    } else if dynamic.type_id == TYPE_INT {
        unsafe { *(dynamic.value_ptr as *const i64) as f64 }
    } else if dynamic.type_id == TYPE_BOOL {
        unsafe {
            if *(dynamic.value_ptr as *const bool) {
                1.0
            } else {
                0.0
            }
        }
    } else {
        0.0
    }
}

/// Unbox a Dynamic as Bool (returns false if wrong type)
#[no_mangle]
pub extern "C" fn haxe_unbox_bool(dynamic: DynamicValue) -> bool {
    if dynamic.type_id == TYPE_BOOL {
        unsafe { *(dynamic.value_ptr as *const bool) }
    } else {
        false
    }
}

/// Unbox a Dynamic as String (returns empty string if wrong type)
#[no_mangle]
pub extern "C" fn haxe_unbox_string(dynamic: DynamicValue) -> StringPtr {
    if dynamic.type_id == TYPE_STRING {
        unsafe { *(dynamic.value_ptr as *const StringPtr) }
    } else {
        StringPtr {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

// ============================================================================
// Std.string() implementation with runtime type dispatch
// ============================================================================

/// Convert a Dynamic value to String using runtime type dispatch
///
/// This is the implementation of Std.string(Dynamic)
#[no_mangle]
pub extern "C" fn haxe_std_string(dynamic: DynamicValue) -> StringPtr {
    // Handle null specially
    if dynamic.type_id == TYPE_NULL || dynamic.value_ptr.is_null() {
        return unsafe { null_to_string(std::ptr::null()) };
    }

    // Look up type info and call toString
    if let Some(type_info) = get_type_info(dynamic.type_id) {
        unsafe { (type_info.to_string)(dynamic.value_ptr) }
    } else {
        // Unknown type, return type name or error
        let s = format!("<unknown type {}>", dynamic.type_id.0);
        let s_static = Box::leak(s.into_boxed_str());
        StringPtr {
            ptr: s_static.as_ptr(),
            len: s_static.len(),
        }
    }
}

/// Convert a Dynamic value to HaxeString pointer using runtime type dispatch
///
/// This is the pointer-returning version of Std.string(Dynamic)
/// Returns *mut HaxeString for proper ABI compatibility
#[no_mangle]
pub extern "C" fn haxe_std_string_ptr(dynamic_ptr: *mut u8) -> *mut crate::haxe_string::HaxeString {
    use crate::haxe_string::HaxeString;

    if dynamic_ptr.is_null() {
        // Return "null" for null pointer
        let s = "null";
        return Box::into_raw(Box::new(HaxeString {
            ptr: s.as_ptr() as *mut u8,
            len: s.len(),
            cap: 0,
        }));
    }

    unsafe {
        let dynamic = *(dynamic_ptr as *const DynamicValue);

        // Handle null type
        if dynamic.type_id == TYPE_NULL || dynamic.value_ptr.is_null() {
            let s = "null";
            return Box::into_raw(Box::new(HaxeString {
                ptr: s.as_ptr() as *mut u8,
                len: s.len(),
                cap: 0,
            }));
        }

        // Look up type info and call toString, then convert to HaxeString
        if let Some(type_info) = get_type_info(dynamic.type_id) {
            let str_ptr = (type_info.to_string)(dynamic.value_ptr);
            // Convert StringPtr to HaxeString (adding cap=0)
            Box::into_raw(Box::new(HaxeString {
                ptr: str_ptr.ptr as *mut u8,
                len: str_ptr.len,
                cap: 0, // StringPtr strings are either static or leaked
            }))
        } else {
            // Unknown type, return type name
            let s = format!("<unknown type {}>", dynamic.type_id.0);
            let bytes = s.into_bytes();
            let len = bytes.len();
            let cap = bytes.capacity();
            let ptr = bytes.as_ptr() as *mut u8;
            std::mem::forget(bytes);
            Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
        }
    }
}

/// Free a Dynamic value
#[no_mangle]
pub extern "C" fn haxe_free_dynamic(dynamic: DynamicValue) {
    if !dynamic.value_ptr.is_null() {
        unsafe {
            libc::free(dynamic.value_ptr as *mut libc::c_void);
        }
    }
}

// ============================================================================
// Std class functions
// ============================================================================

/// Convert a Float to an Int, rounded towards 0
/// Implements Std.int(x:Float):Int
#[no_mangle]
pub extern "C" fn haxe_std_int(x: f64) -> i64 {
    // Truncate towards zero (same as casting in Rust)
    // Handle special cases
    if x.is_nan() {
        return 0;
    }
    if x.is_infinite() {
        if x.is_sign_positive() {
            return i64::MAX;
        } else {
            return i64::MIN;
        }
    }
    x.trunc() as i64
}

/// Parse a String to an Int
/// Implements Std.parseInt(x:String):Null<Int>
/// Returns the parsed value, or i64::MIN as a sentinel for null
/// (caller should check for this and convert to null)
#[no_mangle]
pub extern "C" fn haxe_std_parse_int(str_ptr: *const crate::haxe_string::HaxeString) -> i64 {
    if str_ptr.is_null() {
        return i64::MIN; // Sentinel for null
    }

    let haxe_str = unsafe { &*str_ptr };
    if haxe_str.ptr.is_null() || haxe_str.len == 0 {
        return i64::MIN; // Sentinel for null
    }

    let bytes = unsafe { std::slice::from_raw_parts(haxe_str.ptr, haxe_str.len) };
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return i64::MIN,
    };

    // Trim leading whitespace
    let s = s.trim_start();
    if s.is_empty() {
        return i64::MIN;
    }

    // Handle optional sign
    let (sign, s) = if let Some(rest) = s.strip_prefix('-') {
        (-1i64, rest)
    } else if let Some(rest) = s.strip_prefix('+') {
        (1i64, rest)
    } else {
        (1i64, s)
    };

    // Check for hex prefix (0x or 0X)
    let (radix, s) = if s.starts_with("0x") || s.starts_with("0X") {
        (16, &s[2..])
    } else {
        (10, s)
    };

    // Parse digits until invalid character
    let mut result: i64 = 0;
    let mut has_digits = false;

    for c in s.chars() {
        let digit = match c.to_digit(radix) {
            Some(d) => d as i64,
            None => break, // Stop at invalid character
        };
        has_digits = true;
        result = result.saturating_mul(radix as i64).saturating_add(digit);
    }

    if !has_digits {
        return i64::MIN; // No valid digits found
    }

    sign.saturating_mul(result)
}

/// Parse a String to a Float
/// Implements Std.parseFloat(x:String):Float
/// Returns NaN if parsing fails
#[no_mangle]
pub extern "C" fn haxe_std_parse_float(str_ptr: *const crate::haxe_string::HaxeString) -> f64 {
    if str_ptr.is_null() {
        return f64::NAN;
    }

    let haxe_str = unsafe { &*str_ptr };
    if haxe_str.ptr.is_null() || haxe_str.len == 0 {
        return f64::NAN;
    }

    let bytes = unsafe { std::slice::from_raw_parts(haxe_str.ptr, haxe_str.len) };
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };

    // Trim leading whitespace
    let s = s.trim_start();
    if s.is_empty() {
        return f64::NAN;
    }

    // Try to parse as much as possible
    // Rust's parse is strict, so we need to find the valid prefix
    let mut end_idx = 0;
    let mut _has_dot = false;
    let mut _has_exp = false;
    let chars: Vec<char> = s.chars().collect();

    // Handle optional sign
    if !chars.is_empty() && (chars[0] == '-' || chars[0] == '+') {
        end_idx = 1;
    }

    // Parse integer part
    while end_idx < chars.len() && chars[end_idx].is_ascii_digit() {
        end_idx += 1;
    }

    // Parse decimal part
    if end_idx < chars.len() && chars[end_idx] == '.' {
        _has_dot = true;
        end_idx += 1;
        while end_idx < chars.len() && chars[end_idx].is_ascii_digit() {
            end_idx += 1;
        }
    }

    // Parse exponent part
    if end_idx < chars.len() && (chars[end_idx] == 'e' || chars[end_idx] == 'E') {
        let exp_start = end_idx;
        end_idx += 1;
        if end_idx < chars.len() && (chars[end_idx] == '-' || chars[end_idx] == '+') {
            end_idx += 1;
        }
        let exp_digits_start = end_idx;
        while end_idx < chars.len() && chars[end_idx].is_ascii_digit() {
            end_idx += 1;
        }
        // Only include exponent if there were digits after e/E
        if end_idx == exp_digits_start {
            end_idx = exp_start; // Revert to before 'e'
        } else {
            _has_exp = true;
        }
    }

    if end_idx == 0 || (end_idx == 1 && (chars[0] == '-' || chars[0] == '+')) {
        return f64::NAN;
    }

    let parse_str: String = chars[..end_idx].iter().collect();
    parse_str.parse::<f64>().unwrap_or(f64::NAN)
}

/// Return a random integer between 0 (inclusive) and max (exclusive)
/// Implements Std.random(x:Int):Int
#[no_mangle]
pub extern "C" fn haxe_std_random(max: i64) -> i64 {
    if max <= 1 {
        return 0;
    }

    // Use a simple LCG (Linear Congruential Generator) with thread-local state
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345)
        );
    }

    SEED.with(|seed| {
        // LCG parameters (same as glibc)
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 1 << 31;

        let new_seed = (a.wrapping_mul(seed.get()).wrapping_add(c)) % m;
        seed.set(new_seed);

        (new_seed as i64) % max
    })
}

// ============================================================================
// Pointer-based boxing/unboxing wrappers for MIR (simpler ABI)
// ============================================================================

/// Box an Int as Dynamic (returns opaque pointer to DynamicValue)
#[no_mangle]
pub extern "C" fn haxe_box_int_ptr(value: i64) -> *mut u8 {
    let dynamic = haxe_box_int(value);
    let boxed = Box::new(dynamic);
    Box::into_raw(boxed) as *mut u8
}

/// Box a Float as Dynamic (returns opaque pointer to DynamicValue)
#[no_mangle]
pub extern "C" fn haxe_box_float_ptr(value: f64) -> *mut u8 {
    let dynamic = haxe_box_float(value);
    let boxed = Box::new(dynamic);
    Box::into_raw(boxed) as *mut u8
}

/// Box a Bool as Dynamic (returns opaque pointer to DynamicValue)
#[no_mangle]
pub extern "C" fn haxe_box_bool_ptr(value: bool) -> *mut u8 {
    let dynamic = haxe_box_bool(value);
    let boxed = Box::new(dynamic);
    Box::into_raw(boxed) as *mut u8
}

/// Box a String as Dynamic (returns opaque pointer to DynamicValue)
/// Takes a null-terminated string pointer.
#[no_mangle]
pub extern "C" fn haxe_box_string_ptr(str_ptr: *const u8) -> *mut u8 {
    let len = if str_ptr.is_null() {
        0
    } else {
        unsafe { libc::strlen(str_ptr as *const libc::c_char) }
    };
    let dynamic = haxe_box_string(str_ptr, len);
    let boxed = Box::new(dynamic);
    Box::into_raw(boxed) as *mut u8
}

/// Box a HaxeString pointer as Dynamic.
/// Unlike haxe_box_string_ptr (which expects a null-terminated C string),
/// this takes a pointer to an existing HaxeString struct and wraps it directly.
#[no_mangle]
pub extern "C" fn haxe_box_haxestring_ptr(hs_ptr: *mut u8) -> *mut u8 {
    if hs_ptr.is_null() {
        let dynamic = haxe_box_null();
        let boxed = Box::new(dynamic);
        return Box::into_raw(boxed) as *mut u8;
    }
    let dynamic = DynamicValue {
        type_id: TYPE_STRING,
        value_ptr: hs_ptr,
    };
    let boxed = Box::new(dynamic);
    Box::into_raw(boxed) as *mut u8
}

/// Unbox an Int from Dynamic (takes opaque pointer to DynamicValue)
#[no_mangle]
pub extern "C" fn haxe_unbox_int_ptr(ptr: *mut u8) -> i64 {
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        let dynamic_ptr = ptr as *const DynamicValue;
        let dynamic = *dynamic_ptr;
        haxe_unbox_int(dynamic)
    }
}

/// Unbox a Float from Dynamic (takes opaque pointer to DynamicValue)
#[no_mangle]
pub extern "C" fn haxe_unbox_float_ptr(ptr: *mut u8) -> f64 {
    if ptr.is_null() {
        return 0.0;
    }
    unsafe {
        let dynamic_ptr = ptr as *const DynamicValue;
        let dynamic = *dynamic_ptr;
        haxe_unbox_float(dynamic)
    }
}

/// Unbox a Bool from Dynamic (takes opaque pointer to DynamicValue)
#[no_mangle]
pub extern "C" fn haxe_unbox_bool_ptr(ptr: *mut u8) -> bool {
    if ptr.is_null() {
        return false;
    }
    unsafe {
        let dynamic_ptr = ptr as *const DynamicValue;
        let dynamic = *dynamic_ptr;
        haxe_unbox_bool(dynamic)
    }
}

/// Box a value as Dynamic with a runtime type tag.
/// Used by monomorphized generic code where the concrete type is determined
/// at specialization time via type_param_tag_fixups.
/// Tags: 1=Int, 2=Bool, 4=Float, 5=String, 6=Reference/Object
#[no_mangle]
pub extern "C" fn haxe_box_typed_ptr(value: i64, type_tag: i32) -> *mut u8 {
    match type_tag {
        1 => {
            // Int: allocate and store value, same as haxe_box_int_ptr
            haxe_box_int_ptr(value)
        }
        2 => {
            // Bool: box as bool
            haxe_box_bool_ptr(value != 0)
        }
        4 => {
            // Float: reinterpret bits as f64
            haxe_box_float_ptr(f64::from_bits(value as u64))
        }
        5 => {
            // String: value IS a HaxeString* pointer, store directly as value_ptr
            if value == 0 {
                let dynamic = haxe_box_null();
                let boxed = Box::new(dynamic);
                return Box::into_raw(boxed) as *mut u8;
            }
            let dynamic = DynamicValue {
                type_id: TYPE_STRING,
                value_ptr: value as *mut u8,
            };
            let boxed = Box::new(dynamic);
            Box::into_raw(boxed) as *mut u8
        }
        6 => {
            // Reference type: value is an object pointer, box with generic reference type
            haxe_box_reference_ptr(value as *mut u8, 0)
        }
        _ => {
            // Default: treat as Int
            haxe_box_int_ptr(value)
        }
    }
}

// ============================================================================
// Reference type boxing/unboxing (Classes, Enums, Anonymous, Arrays, etc.)
// ============================================================================

/// Box a reference type (class, enum, anonymous object, array, etc.)
/// The value is already a pointer, so we just wrap it with type metadata
#[no_mangle]
pub extern "C" fn haxe_box_reference_ptr(value_ptr: *mut u8, type_id: u32) -> *mut u8 {
    let dynamic = DynamicValue {
        type_id: TypeId(type_id),
        value_ptr,
    };
    let boxed = Box::new(dynamic);
    Box::into_raw(boxed) as *mut u8
}

/// Unbox a reference type - just extract the pointer
#[no_mangle]
pub extern "C" fn haxe_unbox_reference_ptr(ptr: *mut u8) -> *mut u8 {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    // Debug: Check for suspicious pointer values (like boolean 1 being passed as pointer)
    let ptr_val = ptr as usize;
    if ptr_val < 0x1000 {
        debug!(
            "WARNING: haxe_unbox_reference_ptr received suspicious pointer: {:p} (value={})",
            ptr, ptr_val
        );
        return std::ptr::null_mut();
    }
    unsafe {
        let dynamic_ptr = ptr as *const DynamicValue;
        let dynamic = *dynamic_ptr;
        dynamic.value_ptr
    }
}

// Dynamic arithmetic is handled at the compiler level:
// 1. Unbox both Dynamic operands to f64 via haxe_unbox_float_ptr
// 2. Perform normal MIR arithmetic (FAdd, FSub, etc.)
// 3. Box result via haxe_box_float_ptr (or leave as raw i64 for comparisons)
// This reuses the existing unbox/box infrastructure without dedicated runtime functions.

// ============================================================================
// Object Header Introspection
// ============================================================================

/// Read the runtime type_id from an object's header (first 8 bytes at offset 0).
/// All class instances have a `__type_id: i64` field at GEP index 0.
#[no_mangle]
pub extern "C" fn haxe_object_get_type_id(obj_ptr: *const u8) -> i64 {
    if obj_ptr.is_null() {
        return -1;
    }
    unsafe { *(obj_ptr as *const i64) }
}

/// Safe downcast for class instances using object headers.
/// Reads the type_id from offset 0, walks the class hierarchy, and returns
/// the object pointer on match or null on failure.
/// Used by `cast(expr, Type)` for class→class safe casts.
#[no_mangle]
pub extern "C" fn haxe_safe_downcast_class(obj_ptr: *mut u8, target_type_id: i64) -> *mut u8 {
    if haxe_object_is_instance(obj_ptr, target_type_id) != 0 {
        obj_ptr
    } else {
        std::ptr::null_mut()
    }
}

/// Check if an object is an instance of (or subclass of) a target type.
/// Walks the class hierarchy via TYPE_REGISTRY super_type_id chain.
#[no_mangle]
pub extern "C" fn haxe_object_is_instance(obj_ptr: *const u8, target_type_id: i64) -> i64 {
    if obj_ptr.is_null() {
        return 0;
    }
    let actual_type_id = unsafe { *(obj_ptr as *const i64) };
    if actual_type_id == target_type_id {
        return 1;
    }
    // Walk parent chain via TYPE_REGISTRY
    let registry = TYPE_REGISTRY.read().unwrap();
    if let Some(ref registry) = *registry {
        let mut current = TypeId(actual_type_id as u32);
        while let Some(type_info) = registry.get(&current) {
            if let Some(class_info) = &type_info.class_info {
                if let Some(parent_id) = class_info.super_type_id {
                    if parent_id as i64 == target_type_id {
                        return 1;
                    }
                    current = TypeId(parent_id);
                } else {
                    break; // No parent
                }
            } else {
                break; // Not a class
            }
        }
    }
    0
}

// ============================================================================
// Class Virtual Method Dispatch (Vtable Registry)
// ============================================================================

/// Global vtable registry: type_id (as u32) -> Vec of closure pointers (i64).
/// Each closure pointer points to a `{fn_code_ptr, env_ptr}` struct allocated
/// by `build_function_ref` in the compiler.
static VTABLE_REGISTRY: RwLock<Option<HashMap<u32, Vec<i64>>>> = RwLock::new(None);

/// Initialize a vtable for a class with the given type_id and slot count.
/// Called at program startup before any user code.
#[no_mangle]
pub extern "C" fn haxe_vtable_init(type_id: i32, slot_count: i32) {
    let mut registry = VTABLE_REGISTRY.write().unwrap();
    let map = registry.get_or_insert_with(HashMap::new);
    map.insert(type_id as u32, vec![0i64; slot_count as usize]);
}

/// Store a closure pointer at a vtable slot for a class type_id.
/// The closure_ptr comes from `build_function_ref` — a pointer to
/// `{fn_code_ptr: i64, env_ptr: i64}`.
#[no_mangle]
pub extern "C" fn haxe_vtable_set_slot(type_id: i32, slot_index: i32, closure_ptr: i64) {
    let mut registry = VTABLE_REGISTRY.write().unwrap();
    if let Some(map) = registry.as_mut() {
        if let Some(vtable) = map.get_mut(&(type_id as u32)) {
            if (slot_index as usize) < vtable.len() {
                vtable[slot_index as usize] = closure_ptr;
            }
        }
    }
}

/// Look up a vtable slot for an object. Reads type_id from the object header
/// (first 8 bytes), then returns the closure pointer for the given slot.
/// Returns 0 if no vtable or slot is found.
#[no_mangle]
pub extern "C" fn haxe_vtable_lookup(obj_ptr: *const u8, slot_index: i32) -> i64 {
    if obj_ptr.is_null() {
        return 0;
    }
    let type_id = unsafe { *(obj_ptr as *const i64) } as u32;
    let registry = VTABLE_REGISTRY.read().unwrap();
    if let Some(map) = registry.as_ref() {
        if let Some(vtable) = map.get(&type_id) {
            if (slot_index as usize) < vtable.len() {
                return vtable[slot_index as usize];
            }
        }
    }
    0
}
