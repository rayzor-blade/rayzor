/// Array type implementation using MIR Builder
///
/// Provides array operations with actual MIR function bodies
///
/// Array operations that return Array instances (slice, copy) use out-param
/// convention where the runtime function writes to a provided HaxeArray struct.
/// The MIR wrappers handle allocation and forwarding.
use crate::ir::instructions::CompareOp;
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// HaxeArray runtime structure size in bytes
/// struct HaxeArray { ptr: *mut u8, len: usize, cap: usize, elem_size: usize }
/// On 64-bit: 8 + 8 + 8 + 8 = 32 bytes
const HAXE_ARRAY_STRUCT_SIZE: usize = 32;

/// Iterator object size: __type_id (8) + field1 (8) + field2 (8) = 24 bytes
const ITERATOR_STRUCT_SIZE: usize = 24;

/// Build all array type functions
pub fn build_array_type(builder: &mut MirBuilder) {
    // Declare extern runtime functions
    declare_array_externs(builder);

    // Build MIR wrapper functions
    build_array_push(builder);
    build_array_pop(builder);
    build_array_length(builder);
    build_array_slice(builder);
    build_array_join(builder);
    build_array_index_of(builder);
    build_array_last_index_of(builder);
    build_array_shift(builder);
    build_array_unshift(builder);
    build_array_resize(builder);
    build_array_concat(builder);
    build_array_splice(builder);
    build_array_to_string(builder);
    build_array_map(builder);
    build_array_filter(builder);
    build_array_sort(builder);
    build_array_iterator(builder);
    build_array_kv_iterator(builder);
    build_array_iterator_has_next(builder);
    build_array_iterator_next(builder);
    build_array_kv_iterator_has_next(builder);
    build_array_kv_iterator_next(builder);
}

/// Declare Array extern runtime functions
fn declare_array_externs(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let i64_ty = IrType::I64;
    let _i32_ty = IrType::I32;
    let void_ty = IrType::Void;

    // haxe_array_push_i64(arr: *mut HaxeArray, val: i64)
    let func_id = builder
        .begin_function("haxe_array_push_i64")
        .param("arr", ptr_void.clone())
        .param("val", i64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_pop_ptr(arr: *mut HaxeArray) -> *mut u8
    let func_id = builder
        .begin_function("haxe_array_pop_ptr")
        .param("arr", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_pop_i64(arr: *mut HaxeArray) -> i64
    let func_id = builder
        .begin_function("haxe_array_pop_i64")
        .param("arr", ptr_void.clone())
        .returns(IrType::I64)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_length(arr: *const HaxeArray) -> usize
    let func_id = builder
        .begin_function("haxe_array_length")
        .param("arr", ptr_void.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_slice(out: *mut HaxeArray, arr: *const HaxeArray, start: usize, end: usize)
    let func_id = builder
        .begin_function("haxe_array_slice")
        .param("out", ptr_void.clone())
        .param("arr", ptr_void.clone())
        .param("start", i64_ty.clone())
        .param("end", i64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_copy(out: *mut HaxeArray, arr: *const HaxeArray)
    let func_id = builder
        .begin_function("haxe_array_copy")
        .param("out", ptr_void.clone())
        .param("arr", ptr_void.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_join(arr: *const HaxeArray, sep: *const HaxeString) -> *mut HaxeString
    let func_id = builder
        .begin_function("haxe_array_join")
        .param("arr", ptr_void.clone())
        .param("sep", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_get_f64(arr: *const HaxeArray, index: usize) -> f64
    let func_id = builder
        .begin_function("haxe_array_get_f64")
        .param("arr", ptr_void.clone())
        .param("index", i64_ty.clone())
        .returns(IrType::F64)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_set_i64(arr: *mut HaxeArray, index: usize, value: i64) -> bool
    let func_id = builder
        .begin_function("haxe_array_set_i64")
        .param("arr", ptr_void.clone())
        .param("index", i64_ty.clone())
        .param("value", i64_ty.clone())
        .returns(IrType::Bool)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_set_f64(arr: *mut HaxeArray, index: usize, value: f64) -> bool
    let func_id = builder
        .begin_function("haxe_array_set_f64")
        .param("arr", ptr_void.clone())
        .param("index", i64_ty.clone())
        .param("value", IrType::F64)
        .returns(IrType::Bool)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_set_null(arr: *mut HaxeArray, index: usize) -> bool
    let func_id = builder
        .begin_function("haxe_array_set_null")
        .param("arr", ptr_void.clone())
        .param("index", i64_ty.clone())
        .returns(IrType::Bool)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_index_of(arr: *const HaxeArray, value: i64, from_index: i64) -> i64
    let func_id = builder
        .begin_function("haxe_array_index_of")
        .param("arr", ptr_void.clone())
        .param("value", i64_ty.clone())
        .param("from_index", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_last_index_of(arr: *const HaxeArray, value: i64, from_index: i64) -> i64
    let func_id = builder
        .begin_function("haxe_array_last_index_of")
        .param("arr", ptr_void.clone())
        .param("value", i64_ty.clone())
        .param("from_index", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_contains(arr: *const HaxeArray, value: i64) -> i64
    let func_id = builder
        .begin_function("haxe_array_contains")
        .param("arr", ptr_void.clone())
        .param("value", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_shift(arr: *mut HaxeArray) -> i64
    let func_id = builder
        .begin_function("haxe_array_shift")
        .param("arr", ptr_void.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_shift_ptr(arr: *mut HaxeArray) -> *mut u8 (boxed DynamicValue)
    let func_id = builder
        .begin_function("haxe_array_shift_ptr")
        .param("arr", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_unshift(arr: *mut HaxeArray, value: i64)
    let func_id = builder
        .begin_function("haxe_array_unshift")
        .param("arr", ptr_void.clone())
        .param("value", i64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_resize(arr: *mut HaxeArray, len: i64)
    let func_id = builder
        .begin_function("haxe_array_resize")
        .param("arr", ptr_void.clone())
        .param("len", i64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_concat(out: *mut HaxeArray, arr: *const HaxeArray, other: *const HaxeArray)
    let func_id = builder
        .begin_function("haxe_array_concat")
        .param("out", ptr_void.clone())
        .param("arr", ptr_void.clone())
        .param("other", ptr_void.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_splice(out: *mut HaxeArray, arr: *mut HaxeArray, pos: i64, len: i64)
    let func_id = builder
        .begin_function("haxe_array_splice")
        .param("out", ptr_void.clone())
        .param("arr", ptr_void.clone())
        .param("pos", i64_ty.clone())
        .param("len", i64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_to_string(arr: *const HaxeArray) -> *mut HaxeString
    let func_id = builder
        .begin_function("haxe_array_to_string")
        .param("arr", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_map(out: *mut HaxeArray, arr: *const HaxeArray, fn_ptr: usize, env_ptr: *mut u8)
    let func_id = builder
        .begin_function("haxe_array_map")
        .param("out", ptr_void.clone())
        .param("arr", ptr_void.clone())
        .param("fn_ptr", i64_ty.clone())
        .param("env_ptr", ptr_void.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_filter(out: *mut HaxeArray, arr: *const HaxeArray, fn_ptr: usize, env_ptr: *mut u8)
    let func_id = builder
        .begin_function("haxe_array_filter")
        .param("out", ptr_void.clone())
        .param("arr", ptr_void.clone())
        .param("fn_ptr", i64_ty.clone())
        .param("env_ptr", ptr_void.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_sort(arr: *mut HaxeArray, fn_ptr: usize, env_ptr: *mut u8)
    let func_id = builder
        .begin_function("haxe_array_sort")
        .param("arr", ptr_void.clone())
        .param("fn_ptr", i64_ty.clone())
        .param("env_ptr", ptr_void.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_array_get_i64(arr: *const HaxeArray, index: usize) -> i64
    let func_id = builder
        .begin_function("haxe_array_get_i64")
        .param("arr", ptr_void.clone())
        .param("index", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_anon_new(shape_id: u32, field_count: u32) -> *mut u8
    let func_id = builder
        .begin_function("rayzor_anon_new")
        .param("shape_id", i64_ty.clone())
        .param("field_count", i64_ty.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_ensure_shape(shape_id: u32, descriptor: *mut u8)
    let func_id = builder
        .begin_function("rayzor_ensure_shape")
        .param("shape_id", i64_ty.clone())
        .param("descriptor", ptr_void.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_anon_set_field_by_index(handle: *mut u8, index: u32, value: u64)
    let func_id = builder
        .begin_function("rayzor_anon_set_field_by_index")
        .param("handle", ptr_void.clone())
        .param("index", i64_ty.clone())
        .param("value", i64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_box_reference_ptr(ptr: *mut u8, type_id: u32) -> *mut u8
    // NOTE: Must match runtime signature (2 params). A 1-param declaration here
    // would replace the correct user forward-ref during stdlib merge, causing SIGILL.
    let func_id = builder
        .begin_function("haxe_box_reference_ptr")
        .param("ptr", ptr_void.clone())
        .param("type_id", IrType::U32)
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Build: fn array_push(arr: Any, value: Any) -> void
/// Appends an element to the array
/// Note: Any is represented as i64 in LLVM, matching pointer-sized values
fn build_array_push(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let func_id = builder
        .begin_function("array_push")
        .param("arr", IrType::Any)
        .param("value", IrType::Any)
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let value = builder.get_param(1);

    // Cast arr from Any (i64) to ptr for extern call
    let arr_ptr = builder.cast(arr, IrType::Any, ptr_void);

    // Call runtime function haxe_array_push_i64(arr: *HaxeArray, val: i64)
    // value is already i64 (Any), which matches haxe_array_push_i64's signature
    let extern_func = builder
        .get_function_by_name("haxe_array_push_i64")
        .expect("haxe_array_push_i64 extern not found");

    builder.call(extern_func, vec![arr_ptr, value]);

    builder.ret(None);
}

/// Build: fn array_pop(arr: Any) -> Any
/// Removes and returns the last element from the array
fn build_array_pop(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let func_id = builder
        .begin_function("array_pop")
        .param("arr", IrType::Any)
        .returns(IrType::I64)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);

    // Cast arr from Any (i64) to ptr for extern call
    let arr_ptr = builder.cast(arr, IrType::Any, ptr_void.clone());

    // Call haxe_array_pop_i64 which returns the raw i64 value (no boxing).
    // For class-typed arrays, the i64 IS the raw class pointer.
    // The hir_to_mir.rs code casts I64→Ptr(Void) when needed.
    let extern_func = builder
        .get_function_by_name("haxe_array_pop_i64")
        .expect("haxe_array_pop_i64 extern not found");

    if let Some(result) = builder.call(extern_func, vec![arr_ptr]) {
        builder.ret(Some(result));
    } else {
        let zero_val = builder.const_value(crate::ir::IrValue::I64(0));
        builder.ret(Some(zero_val));
    }
}

/// Build: fn array_length(arr: Any) -> i64
/// Returns the length of the array (usize as i64)
fn build_array_length(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let func_id = builder
        .begin_function("array_length")
        .param("arr", IrType::Any)
        .returns(IrType::I64)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);

    // Cast arr from Any (i64) to ptr for extern call
    let arr_ptr = builder.cast(arr, IrType::Any, ptr_void);

    // Call runtime function haxe_array_length(arr: *HaxeArray) -> i64 (usize)
    let extern_func = builder
        .get_function_by_name("haxe_array_length")
        .expect("haxe_array_length extern not found");

    if let Some(len_i64) = builder.call(extern_func, vec![arr_ptr]) {
        // Return i64 directly - no cast needed
        builder.ret(Some(len_i64));
    } else {
        let zero = builder.const_i64(0);
        builder.ret(Some(zero));
    }
}

/// Build: fn array_slice(arr: Ptr(Void), start: i64, end: i64) -> Ptr(Void)
/// Wrapper for haxe_array_slice that handles out-param allocation
///
/// This wrapper:
/// 1. Allocates space for the result HaxeArray struct (32 bytes)
/// 2. Calls haxe_array_slice(out_ptr, arr, start, end)
/// 3. Returns the pointer to the allocated result
fn build_array_slice(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let i64_ty = IrType::I64;

    // Function signature: array_slice(arr: *Array, start: i64, end: i64) -> *Array
    let func_id = builder
        .begin_function("array_slice")
        .param("arr", ptr_void.clone())
        .param("start", i64_ty.clone())
        .param("end", i64_ty.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let start = builder.get_param(1);
    let end = builder.get_param(2);

    // HEAP-allocate space for HaxeArray struct (32 bytes)
    // HaxeArray struct: { ptr: *mut u8, len: usize, cap: usize, elem_size: usize }
    // Must use heap allocation since we're returning this pointer!
    let malloc_func = builder
        .get_function_by_name("malloc")
        .expect("malloc extern not found");
    let size = builder.const_i64(HAXE_ARRAY_STRUCT_SIZE as i64);
    let out_ptr = builder
        .call(malloc_func, vec![size])
        .expect("malloc should return a pointer");

    // Call haxe_array_slice(out_ptr, arr, start, end)
    let slice_func = builder
        .get_function_by_name("haxe_array_slice")
        .expect("haxe_array_slice extern not found");

    builder.call(slice_func, vec![out_ptr, arr, start, end]);

    // Return the pointer to the heap-allocated array
    builder.ret(Some(out_ptr));
}

/// Build: fn array_join(arr: Ptr(Void), sep: Ptr(Void)) -> Ptr(Void)
/// Wrapper for haxe_array_join that joins array elements with separator
fn build_array_join(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));

    // Function signature: array_join(arr: *Array, sep: *String) -> *String
    let func_id = builder
        .begin_function("array_join")
        .param("arr", ptr_void.clone())
        .param("sep", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let sep = builder.get_param(1);

    // Call haxe_array_join(arr, sep) -> *String
    let join_func = builder
        .get_function_by_name("haxe_array_join")
        .expect("haxe_array_join extern not found");

    if let Some(result) = builder.call(join_func, vec![arr, sep]) {
        builder.ret(Some(result));
    } else {
        // Return null on failure
        let null_val = builder.const_value(crate::ir::IrValue::Null);
        builder.ret(Some(null_val));
    }
}

/// Build: fn array_index_of(arr: Ptr(Void), value: i64) -> i64
/// Wrapper for haxe_array_index_of that defaults fromIndex to 0
fn build_array_index_of(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("array_index_of")
        .param("arr", ptr_void.clone())
        .param("value", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let value = builder.get_param(1);
    let from_index = builder.const_i64(0);

    let index_of_func = builder
        .get_function_by_name("haxe_array_index_of")
        .expect("haxe_array_index_of extern not found");

    if let Some(result) = builder.call(index_of_func, vec![arr, value, from_index]) {
        builder.ret(Some(result));
    } else {
        let neg_one = builder.const_i64(-1);
        builder.ret(Some(neg_one));
    }
}

/// Build: fn array_last_index_of(arr: Ptr(Void), value: i64) -> i64
/// Wrapper for haxe_array_last_index_of that defaults fromIndex to i64::MAX (search from end)
fn build_array_last_index_of(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("array_last_index_of")
        .param("arr", ptr_void.clone())
        .param("value", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let value = builder.get_param(1);
    // i64::MAX as sentinel — runtime handles "from end" when >= len
    let from_index = builder.const_i64(i64::MAX);

    let last_index_of_func = builder
        .get_function_by_name("haxe_array_last_index_of")
        .expect("haxe_array_last_index_of extern not found");

    if let Some(result) = builder.call(last_index_of_func, vec![arr, value, from_index]) {
        builder.ret(Some(result));
    } else {
        let neg_one = builder.const_i64(-1);
        builder.ret(Some(neg_one));
    }
}

/// Build: fn array_shift(arr: Any) -> Any
/// Wrapper: removes and returns first element as boxed DynamicValue*
fn build_array_shift(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("array_shift")
        .param("arr", IrType::Any)
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let arr_ptr = builder.cast(arr, IrType::Any, ptr_void.clone());

    // Call haxe_array_shift_ptr which returns boxed DynamicValue*
    let shift_func = builder
        .get_function_by_name("haxe_array_shift_ptr")
        .expect("haxe_array_shift_ptr extern not found");

    if let Some(result) = builder.call(shift_func, vec![arr_ptr]) {
        builder.ret(Some(result));
    } else {
        let null_val = builder.const_value(crate::ir::IrValue::Null);
        builder.ret(Some(null_val));
    }
}

/// Build: fn array_unshift(arr: Any, value: Any) -> void
/// Wrapper: adds element at the beginning
fn build_array_unshift(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));

    let func_id = builder
        .begin_function("array_unshift")
        .param("arr", IrType::Any)
        .param("value", IrType::Any)
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let value = builder.get_param(1);
    let arr_ptr = builder.cast(arr, IrType::Any, ptr_void);

    let unshift_func = builder
        .get_function_by_name("haxe_array_unshift")
        .expect("haxe_array_unshift extern not found");

    builder.call(unshift_func, vec![arr_ptr, value]);
    builder.ret(None);
}

/// Build: fn array_resize(arr: Any, len: i64) -> void
/// Wrapper: sets array length
fn build_array_resize(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));

    let func_id = builder
        .begin_function("array_resize")
        .param("arr", IrType::Any)
        .param("len", IrType::I64)
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let len = builder.get_param(1);
    let arr_ptr = builder.cast(arr, IrType::Any, ptr_void);

    let resize_func = builder
        .get_function_by_name("haxe_array_resize")
        .expect("haxe_array_resize extern not found");

    builder.call(resize_func, vec![arr_ptr, len]);
    builder.ret(None);
}

/// Build: fn array_concat(arr: Ptr(Void), other: Ptr(Void)) -> Ptr(Void)
/// Wrapper for haxe_array_concat that handles out-param allocation
fn build_array_concat(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));

    let func_id = builder
        .begin_function("array_concat")
        .param("arr", ptr_void.clone())
        .param("other", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let other = builder.get_param(1);

    // Allocate out array struct (32 bytes)
    let malloc_func = builder
        .get_function_by_name("malloc")
        .expect("malloc extern not found");
    let size = builder.const_i64(HAXE_ARRAY_STRUCT_SIZE as i64);
    let out_ptr = builder
        .call(malloc_func, vec![size])
        .expect("malloc should return a pointer");

    // Call haxe_array_concat(out, arr, other)
    let concat_func = builder
        .get_function_by_name("haxe_array_concat")
        .expect("haxe_array_concat extern not found");
    builder.call(concat_func, vec![out_ptr, arr, other]);

    builder.ret(Some(out_ptr));
}

/// Build: fn array_splice(arr: Ptr(Void), pos: i64, len: i64) -> Ptr(Void)
/// Wrapper for haxe_array_splice that handles out-param allocation
fn build_array_splice(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("array_splice")
        .param("arr", ptr_void.clone())
        .param("pos", i64_ty.clone())
        .param("len", i64_ty)
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let pos = builder.get_param(1);
    let len = builder.get_param(2);

    // Allocate out array struct (32 bytes)
    let malloc_func = builder
        .get_function_by_name("malloc")
        .expect("malloc extern not found");
    let size = builder.const_i64(HAXE_ARRAY_STRUCT_SIZE as i64);
    let out_ptr = builder
        .call(malloc_func, vec![size])
        .expect("malloc should return a pointer");

    // Call haxe_array_splice(out, arr, pos, len)
    let splice_func = builder
        .get_function_by_name("haxe_array_splice")
        .expect("haxe_array_splice extern not found");
    builder.call(splice_func, vec![out_ptr, arr, pos, len]);

    builder.ret(Some(out_ptr));
}

/// Build: fn array_to_string(arr: Ptr(Void)) -> Ptr(Void)
/// Wrapper for haxe_array_to_string
fn build_array_to_string(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));

    let func_id = builder
        .begin_function("array_to_string")
        .param("arr", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);

    // Call haxe_array_to_string(arr) -> *HaxeString
    let to_string_func = builder
        .get_function_by_name("haxe_array_to_string")
        .expect("haxe_array_to_string extern not found");

    if let Some(result) = builder.call(to_string_func, vec![arr]) {
        builder.ret(Some(result));
    } else {
        let null_val = builder.const_value(crate::ir::IrValue::Null);
        builder.ret(Some(null_val));
    }
}

/// Build: fn array_map(arr: Any, closure: Any) -> Ptr(Void)
/// Applies callback to each element, returns new array.
/// Closure struct layout: { fn_ptr: i64, env_ptr: i64 }
fn build_array_map(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("array_map")
        .param("arr", IrType::Any)
        .param("closure", IrType::Any)
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let closure = builder.get_param(1);

    // Cast arr from Any to Ptr for extern call
    let arr_ptr = builder.cast(arr, IrType::Any, ptr_void.clone());

    // Cast closure from Any to Ptr to load fields
    let closure_ptr = builder.cast(closure, IrType::Any, ptr_u8.clone());

    // Load fn_ptr from closure[0]
    let fn_ptr = builder.load(closure_ptr, IrType::I64);

    // Load env_ptr from closure[8]
    let offset_8 = builder.const_i64(8);
    let env_slot = builder.ptr_add(closure_ptr, offset_8, ptr_u8.clone());
    let env_ptr = builder.load(env_slot, IrType::I64);
    let env_ptr_cast = builder.cast(env_ptr, IrType::I64, ptr_void.clone());

    // Allocate out array struct (32 bytes)
    let malloc_func = builder
        .get_function_by_name("malloc")
        .expect("malloc extern not found");
    let size = builder.const_i64(HAXE_ARRAY_STRUCT_SIZE as i64);
    let out_ptr = builder
        .call(malloc_func, vec![size])
        .expect("malloc should return a pointer");

    // Call haxe_array_map(out, arr, fn_ptr, env_ptr)
    let map_func = builder
        .get_function_by_name("haxe_array_map")
        .expect("haxe_array_map extern not found");
    builder.call(map_func, vec![out_ptr, arr_ptr, fn_ptr, env_ptr_cast]);

    builder.ret(Some(out_ptr));
}

/// Build: fn array_filter(arr: Any, closure: Any) -> Ptr(Void)
/// Keeps elements where callback returns true, returns new array.
fn build_array_filter(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("array_filter")
        .param("arr", IrType::Any)
        .param("closure", IrType::Any)
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let closure = builder.get_param(1);

    let arr_ptr = builder.cast(arr, IrType::Any, ptr_void.clone());
    let closure_ptr = builder.cast(closure, IrType::Any, ptr_u8.clone());

    let fn_ptr = builder.load(closure_ptr, IrType::I64);

    let offset_8 = builder.const_i64(8);
    let env_slot = builder.ptr_add(closure_ptr, offset_8, ptr_u8.clone());
    let env_ptr = builder.load(env_slot, IrType::I64);
    let env_ptr_cast = builder.cast(env_ptr, IrType::I64, ptr_void.clone());

    let malloc_func = builder
        .get_function_by_name("malloc")
        .expect("malloc extern not found");
    let size = builder.const_i64(HAXE_ARRAY_STRUCT_SIZE as i64);
    let out_ptr = builder
        .call(malloc_func, vec![size])
        .expect("malloc should return a pointer");

    let filter_func = builder
        .get_function_by_name("haxe_array_filter")
        .expect("haxe_array_filter extern not found");
    builder.call(filter_func, vec![out_ptr, arr_ptr, fn_ptr, env_ptr_cast]);

    builder.ret(Some(out_ptr));
}

/// Build: fn array_iterator(arr: Ptr(Void)) -> Ptr(Void)
/// Creates an ArrayIterator object with correct compiled class layout:
///   offset 0:  __type_id (i64) = 0 (placeholder)
///   offset 8:  array (Ptr) = arr parameter
///   offset 16: current (i64) = 0
/// The returned object is compatible with compiled ArrayIterator.hx methods (hasNext, next).
fn build_array_iterator(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("array_iterator")
        .param("arr", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);

    // Allocate iterator struct (24 bytes: type_id + array + current)
    let malloc_func = builder
        .get_function_by_name("malloc")
        .expect("malloc extern not found");
    let size = builder.const_i64(ITERATOR_STRUCT_SIZE as i64);
    let ptr = builder
        .call(malloc_func, vec![size])
        .expect("malloc should return a pointer");

    // offset 0: type_id = 0 (placeholder)
    let zero = builder.const_i64(0);
    builder.store(ptr, zero);

    // offset 8: array pointer
    let off8 = builder.const_i64(8);
    let slot1 = builder.ptr_add(ptr, off8, ptr_u8.clone());
    builder.store(slot1, arr);

    // offset 16: current = 0
    let zero2 = builder.const_i64(0);
    let off16 = builder.const_i64(16);
    let slot2 = builder.ptr_add(ptr, off16, ptr_u8.clone());
    builder.store(slot2, zero2);

    // Return the iterator pointer
    let result = builder.cast(ptr, ptr_u8, ptr_void);
    builder.ret(Some(result));
}

/// Build: fn array_kv_iterator(arr: Ptr(Void)) -> Ptr(Void)
/// Creates an ArrayKeyValueIterator object with correct compiled class layout:
///   offset 0:  __type_id (i64) = 0 (placeholder)
///   offset 8:  current (i64) = 0   (declared first in ArrayKeyValueIterator.hx)
///   offset 16: array (Ptr) = arr   (declared second)
/// The returned object is compatible with compiled ArrayKeyValueIterator.hx methods.
fn build_array_kv_iterator(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("array_kv_iterator")
        .param("arr", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);

    // Allocate iterator struct (24 bytes: type_id + current + array)
    let malloc_func = builder
        .get_function_by_name("malloc")
        .expect("malloc extern not found");
    let size = builder.const_i64(ITERATOR_STRUCT_SIZE as i64);
    let ptr = builder
        .call(malloc_func, vec![size])
        .expect("malloc should return a pointer");

    // offset 0: type_id = 0 (placeholder)
    let zero = builder.const_i64(0);
    builder.store(ptr, zero);

    // offset 8: current = 0 (declared first in ArrayKeyValueIterator.hx)
    let zero2 = builder.const_i64(0);
    let off8 = builder.const_i64(8);
    let slot1 = builder.ptr_add(ptr, off8, ptr_u8.clone());
    builder.store(slot1, zero2);

    // offset 16: array pointer (declared second in ArrayKeyValueIterator.hx)
    let off16 = builder.const_i64(16);
    let slot2 = builder.ptr_add(ptr, off16, ptr_u8.clone());
    builder.store(slot2, arr);

    // Return the iterator pointer
    let result = builder.cast(ptr, ptr_u8, ptr_void);
    builder.ret(Some(result));
}

/// Build: fn ArrayIterator_hasNext(iter: Ptr(Void)) -> I32
/// Checks if the ArrayIterator has more elements.
/// Layout: [type_id(0), array(8), current(16)]
fn build_array_iterator_has_next(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("ArrayIterator_hasNext")
        .param("iter", ptr_void.clone())
        .returns(IrType::I32)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let iter = builder.get_param(0);

    // Load array pointer from offset 8
    let off8 = builder.const_i64(8);
    let array_slot = builder.ptr_add(iter, off8, ptr_u8.clone());
    let array_ptr = builder.load(array_slot, ptr_void.clone());

    // Load current from offset 16
    let off16 = builder.const_i64(16);
    let current_slot = builder.ptr_add(iter, off16, ptr_u8.clone());
    let current = builder.load(current_slot, IrType::I64);

    // Call haxe_array_length(array) -> I64
    let length_func = builder
        .get_function_by_name("haxe_array_length")
        .expect("haxe_array_length extern not found");
    let length = builder
        .call(length_func, vec![array_ptr])
        .expect("haxe_array_length should return");

    // Compare current < length
    let is_less = builder.cmp(CompareOp::Lt, current, length);

    // Cast Bool to I32
    let result = builder.cast(is_less, IrType::Bool, IrType::I32);
    builder.ret(Some(result));
}

/// Build: fn ArrayIterator_next(iter: Ptr(Void)) -> I64
/// Returns the next element and advances the iterator.
/// Layout: [type_id(0), array(8), current(16)]
fn build_array_iterator_next(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("ArrayIterator_next")
        .param("iter", ptr_void.clone())
        .returns(IrType::I64)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let iter = builder.get_param(0);

    // Load array pointer from offset 8
    let off8 = builder.const_i64(8);
    let array_slot = builder.ptr_add(iter, off8, ptr_u8.clone());
    let array_ptr = builder.load(array_slot, ptr_void.clone());

    // Load current from offset 16
    let off16 = builder.const_i64(16);
    let current_slot = builder.ptr_add(iter, off16, ptr_u8.clone());
    let current = builder.load(current_slot, IrType::I64);

    // Call haxe_array_get_i64(array, current) -> I64
    let get_func = builder
        .get_function_by_name("haxe_array_get_i64")
        .expect("haxe_array_get_i64 extern not found");
    let value = builder
        .call(get_func, vec![array_ptr, current])
        .expect("haxe_array_get_i64 should return");

    // Increment current: current + 1
    let one = builder.const_i64(1);
    let new_current = builder.add(current, one, IrType::I64);

    // Store new_current back at offset 16
    let current_slot2 = builder.ptr_add(iter, off16, ptr_u8.clone());
    builder.store(current_slot2, new_current);

    // Return the value
    builder.ret(Some(value));
}

/// Build: fn ArrayKeyValueIterator_hasNext(iter: Ptr(Void)) -> I32
/// Checks if the ArrayKeyValueIterator has more elements.
/// Layout: [type_id(0), current(8), array(16)]
fn build_array_kv_iterator_has_next(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("ArrayKeyValueIterator_hasNext")
        .param("iter", ptr_void.clone())
        .returns(IrType::I32)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let iter = builder.get_param(0);

    // Load current from offset 8
    let off8 = builder.const_i64(8);
    let current_slot = builder.ptr_add(iter, off8, ptr_u8.clone());
    let current = builder.load(current_slot, IrType::I64);

    // Load array pointer from offset 16
    let off16 = builder.const_i64(16);
    let array_slot = builder.ptr_add(iter, off16, ptr_u8.clone());
    let array_ptr = builder.load(array_slot, ptr_void.clone());

    // Call haxe_array_length(array) -> I64
    let length_func = builder
        .get_function_by_name("haxe_array_length")
        .expect("haxe_array_length extern not found");
    let length = builder
        .call(length_func, vec![array_ptr])
        .expect("haxe_array_length should return");

    // Compare current < length
    let is_less = builder.cmp(CompareOp::Lt, current, length);

    // Cast Bool to I32
    let result = builder.cast(is_less, IrType::Bool, IrType::I32);
    builder.ret(Some(result));
}

/// Build: fn ArrayKeyValueIterator_next(iter: Ptr(Void)) -> Ptr(Void)
/// Returns the next {key: Int, value: Dynamic} anon object and advances the iterator.
/// Layout: [type_id(0), current(8), array(16)]
fn build_array_kv_iterator_next(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("ArrayKeyValueIterator_next")
        .param("iter", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let iter = builder.get_param(0);

    // Load current from offset 8
    let off8 = builder.const_i64(8);
    let current_slot = builder.ptr_add(iter, off8, ptr_u8.clone());
    let current = builder.load(current_slot, IrType::I64);

    // Load array pointer from offset 16
    let off16 = builder.const_i64(16);
    let array_slot = builder.ptr_add(iter, off16, ptr_u8.clone());
    let array_ptr = builder.load(array_slot, ptr_void.clone());

    // Call haxe_array_get_i64(array, current) -> I64
    let get_func = builder
        .get_function_by_name("haxe_array_get_i64")
        .expect("haxe_array_get_i64 extern not found");
    let value = builder
        .call(get_func, vec![array_ptr, current])
        .expect("haxe_array_get_i64 should return");

    // Create anon object {key: current, value: element}
    // KV_SHAPE_ID = 1001, fields sorted: key(idx 0), value(idx 1)
    let anon_new_func = builder
        .get_function_by_name("rayzor_anon_new")
        .expect("rayzor_anon_new extern not found");
    let ensure_shape_func = builder
        .get_function_by_name("rayzor_ensure_shape")
        .expect("rayzor_ensure_shape extern not found");
    let set_field_func = builder
        .get_function_by_name("rayzor_anon_set_field_by_index")
        .expect("rayzor_anon_set_field_by_index extern not found");

    // Ensure shape {key:3,value:7} is registered
    let shape_id = builder.const_i64(1001); // KV_SHAPE_ID from runtime
                                            // We skip ensure_shape here — the runtime ArrayKeyValueIterator already registers it,
                                            // and the shape is lazily created by rayzor_anon_new if needed.

    let field_count = builder.const_i64(2);
    let handle = builder
        .call(anon_new_func, vec![shape_id, field_count])
        .expect("rayzor_anon_new should return");

    // Set key field (index 0)
    let idx0 = builder.const_i64(0);
    let key_as_u64 = builder.cast(current, IrType::I64, IrType::I64); // current is already I64
    builder.call(set_field_func, vec![handle, idx0, key_as_u64]);

    // Set value field (index 1)
    let idx1 = builder.const_i64(1);
    let val_as_u64 = builder.cast(value, IrType::I64, IrType::I64);
    builder.call(set_field_func, vec![handle, idx1, val_as_u64]);

    // Increment current: current + 1
    let one = builder.const_i64(1);
    let new_current = builder.add(current, one, IrType::I64);

    // Store new_current back at offset 8
    let current_slot2 = builder.ptr_add(iter, off8, ptr_u8.clone());
    builder.store(current_slot2, new_current);

    // Box the anon handle as DynamicValue so Dynamic field access can unbox it correctly
    let box_func = builder
        .get_function_by_name("haxe_box_reference_ptr")
        .expect("haxe_box_reference_ptr extern not found");
    // type_id=0 for anonymous objects (no specific class type)
    let type_id = builder.const_i64(0);
    let type_id_u32 = builder.cast(type_id, IrType::I64, IrType::U32);
    let boxed = builder
        .call(box_func, vec![handle, type_id_u32])
        .expect("haxe_box_reference_ptr should return");

    builder.ret(Some(boxed));
}

/// Build: fn array_sort(arr: Any, closure: Any) -> Void
/// Sorts array in-place using comparator callback.
fn build_array_sort(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("array_sort")
        .param("arr", IrType::Any)
        .param("closure", IrType::Any)
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);
    let closure = builder.get_param(1);

    let arr_ptr = builder.cast(arr, IrType::Any, ptr_void.clone());
    let closure_ptr = builder.cast(closure, IrType::Any, ptr_u8.clone());

    let fn_ptr = builder.load(closure_ptr, IrType::I64);

    let offset_8 = builder.const_i64(8);
    let env_slot = builder.ptr_add(closure_ptr, offset_8, ptr_u8.clone());
    let env_ptr = builder.load(env_slot, IrType::I64);
    let env_ptr_cast = builder.cast(env_ptr, IrType::I64, ptr_void.clone());

    let sort_func = builder
        .get_function_by_name("haxe_array_sort")
        .expect("haxe_array_sort extern not found");
    builder.call(sort_func, vec![arr_ptr, fn_ptr, env_ptr_cast]);

    builder.ret(None);
}
