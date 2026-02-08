/// Array type implementation using MIR Builder
///
/// Provides array operations with actual MIR function bodies
///
/// Array operations that return Array instances (slice, copy) use out-param
/// convention where the runtime function writes to a provided HaxeArray struct.
/// The MIR wrappers handle allocation and forwarding.
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// HaxeArray runtime structure size in bytes
/// struct HaxeArray { ptr: *mut u8, len: usize, cap: usize, elem_size: usize }
/// On 64-bit: 8 + 8 + 8 + 8 = 32 bytes
const HAXE_ARRAY_STRUCT_SIZE: usize = 32;

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
    build_array_map(builder);
    build_array_filter(builder);
    build_array_sort(builder);
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
        .returns(IrType::Any)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arr = builder.get_param(0);

    // Cast arr from Any (i64) to ptr for extern call
    let arr_ptr = builder.cast(arr, IrType::Any, ptr_void.clone());

    // Call runtime function haxe_array_pop_ptr(arr: *HaxeArray) -> *mut u8
    let extern_func = builder
        .get_function_by_name("haxe_array_pop_ptr")
        .expect("haxe_array_pop_ptr extern not found");

    if let Some(result) = builder.call(extern_func, vec![arr_ptr]) {
        // Cast result from ptr to Any (i64)
        let result_i64 = builder.cast(result, ptr_void, IrType::Any);
        builder.ret(Some(result_i64));
    } else {
        let null_val = builder.const_value(crate::ir::IrValue::Null);
        builder.ret(Some(null_val));
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
