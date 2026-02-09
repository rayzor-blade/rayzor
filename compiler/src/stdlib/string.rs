/// String type implementation using MIR Builder
///
/// Provides string operations with actual MIR function bodies
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// Build all string type functions
pub fn build_string_type(builder: &mut MirBuilder) {
    // Declare extern functions first
    declare_string_externs(builder);

    build_string_new(builder);
    build_string_concat(builder);
    build_trace(builder);

    // MIR wrapper functions for String methods with optional parameters
    // 1-arg versions provide default startIndex
    build_string_indexof_wrapper(builder);
    build_string_lastindexof_wrapper(builder);
    // 2-arg versions pass through the explicit startIndex
    build_string_indexof_2_wrapper(builder);
    build_string_lastindexof_2_wrapper(builder);

    // MIR wrappers for charAt and substring
    build_string_charat_wrapper(builder);
    build_string_substring_wrapper(builder);
}

/// Declare extern runtime functions for string operations
fn declare_string_externs(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
    let i32_ty = IrType::I32;

    // extern fn haxe_string_concat(a: *String, b: *String) -> *String
    // Returns a pointer to avoid struct return ABI issues
    let func_id = builder
        .begin_function("haxe_string_concat")
        .param("a", string_ptr_ty.clone())
        .param("b", string_ptr_ty.clone())
        .returns(string_ptr_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn haxe_string_split_array(s: *String, delim: *String) -> *HaxeArray
    // Returns a proper HaxeArray structure containing string pointers
    let func_id = builder
        .begin_function("haxe_string_split_array")
        .param("s", string_ptr_ty.clone())
        .param("delimiter", string_ptr_ty.clone())
        .returns(ptr_void.clone()) // Returns *HaxeArray
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn haxe_string_index_of_ptr(s: *String, needle: *String, startIndex: i32) -> i32
    let func_id = builder
        .begin_function("haxe_string_index_of_ptr")
        .param("s", string_ptr_ty.clone())
        .param("needle", string_ptr_ty.clone())
        .param("start_index", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn haxe_string_last_index_of_ptr(s: *String, needle: *String, startIndex: i32) -> i32
    let func_id = builder
        .begin_function("haxe_string_last_index_of_ptr")
        .param("s", string_ptr_ty.clone())
        .param("needle", string_ptr_ty.clone())
        .param("start_index", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn haxe_string_char_at_ptr(s: *String, index: i32) -> *String
    // Returns a single-character string at the given index
    let func_id = builder
        .begin_function("haxe_string_char_at_ptr")
        .param("s", string_ptr_ty.clone())
        .param("index", i32_ty.clone())
        .returns(string_ptr_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn haxe_string_substring_ptr(s: *String, startIndex: i32, endIndex: i32) -> *String
    // Returns a substring from startIndex to endIndex (exclusive)
    let func_id = builder
        .begin_function("haxe_string_substring_ptr")
        .param("s", string_ptr_ty.clone())
        .param("start_index", i32_ty.clone())
        .param("end_index", i32_ty.clone())
        .returns(string_ptr_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn haxe_string_substr_ptr(s: *String, startIndex: i32, length: i32) -> *String
    // Returns a substring starting at startIndex with given length
    let func_id = builder
        .begin_function("haxe_string_substr_ptr")
        .param("s", string_ptr_ty.clone())
        .param("start_index", i32_ty.clone())
        .param("length", i32_ty.clone())
        .returns(string_ptr_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Build: fn string_new() -> String
/// Creates an empty string
fn build_string_new(builder: &mut MirBuilder) {
    let func_id = builder
        .begin_function("string_new")
        .returns(IrType::String)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Return an empty string constant
    let empty_str = builder.const_string("");
    builder.ret(Some(empty_str));
}

/// Build: fn string_concat(s1: *String, s2: *String) -> *String
/// Concatenates two strings by calling the runtime function
/// Returns a pointer to avoid struct return ABI issues
fn build_string_concat(builder: &mut MirBuilder) {
    let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));

    let func_id = builder
        .begin_function("string_concat")
        .param("s1", string_ptr_ty.clone())
        .param("s2", string_ptr_ty.clone())
        .returns(string_ptr_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s1_ptr = builder.get_param(0);
    let s2_ptr = builder.get_param(1);

    // Call extern runtime function directly
    let extern_id = builder
        .get_function_by_name("haxe_string_concat")
        .expect("haxe_string_concat not found");
    let result = builder.call(extern_id, vec![s1_ptr, s2_ptr]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn trace(value: &String) -> void
/// Prints a value to the console (Haxe's trace function)
fn build_trace(builder: &mut MirBuilder) {
    let string_ref_ty = IrType::Ref(Box::new(IrType::String));
    let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));

    // First, declare the extern trace function if not already declared
    let extern_id = builder
        .begin_function("haxe_trace_string_struct")
        .param("s", string_ptr_ty.clone())
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(extern_id);

    // Now build the wrapper function
    let func_id = builder
        .begin_function("trace")
        .param("value", string_ref_ty.clone())
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .public()
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let value_ptr = builder.get_param(0);

    // Call the extern runtime function to actually print
    let trace_func = builder
        .get_function_by_name("haxe_trace_string_struct")
        .expect("haxe_trace_string_struct not found");
    let _ = builder.call(trace_func, vec![value_ptr]);

    builder.ret(None);
}

/// Build: fn String_indexOf(s: *String, needle: *String) -> i32
/// MIR wrapper for String.indexOf that provides default startIndex=0
/// This is the method called when user writes: s.indexOf(needle)
fn build_string_indexof_wrapper(builder: &mut MirBuilder) {
    let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("String_indexOf")
        .param("s", string_ptr_ty.clone())
        .param("needle", string_ptr_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s = builder.get_param(0);
    let needle = builder.get_param(1);

    // Default startIndex = 0
    let start_index = builder.const_i32(0);

    // Call the runtime function with default startIndex
    let extern_id = builder
        .get_function_by_name("haxe_string_index_of_ptr")
        .expect("haxe_string_index_of_ptr not found");
    let result = builder
        .call(extern_id, vec![s, needle, start_index])
        .unwrap();

    builder.ret(Some(result));
}

/// Build: fn String_lastIndexOf(s: *String, needle: *String) -> i32
/// MIR wrapper for String.lastIndexOf that provides default startIndex=-1 (search from end)
/// This is the method called when user writes: s.lastIndexOf(needle)
fn build_string_lastindexof_wrapper(builder: &mut MirBuilder) {
    let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("String_lastIndexOf")
        .param("s", string_ptr_ty.clone())
        .param("needle", string_ptr_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s = builder.get_param(0);
    let needle = builder.get_param(1);

    // Default startIndex = -1 (means search from end in runtime)
    // Actually for lastIndexOf, the default should be null which means string.length
    // In Haxe, passing -1 or a large number would search from end
    let start_index = builder.const_i32(-1);

    // Call the runtime function with default startIndex
    let extern_id = builder
        .get_function_by_name("haxe_string_last_index_of_ptr")
        .expect("haxe_string_last_index_of_ptr not found");
    let result = builder
        .call(extern_id, vec![s, needle, start_index])
        .unwrap();

    builder.ret(Some(result));
}

/// Build: fn String_indexOf_2(s: *String, needle: *String, startIndex: i32) -> i32
/// MIR wrapper for String.indexOf with explicit startIndex
/// This is used when user writes: s.indexOf(needle, startIndex)
fn build_string_indexof_2_wrapper(builder: &mut MirBuilder) {
    let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("String_indexOf_2")
        .param("s", string_ptr_ty.clone())
        .param("needle", string_ptr_ty.clone())
        .param("start_index", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s = builder.get_param(0);
    let needle = builder.get_param(1);
    let start_index = builder.get_param(2);

    // Forward to the runtime function
    let extern_id = builder
        .get_function_by_name("haxe_string_index_of_ptr")
        .expect("haxe_string_index_of_ptr not found");
    let result = builder
        .call(extern_id, vec![s, needle, start_index])
        .unwrap();

    builder.ret(Some(result));
}

/// Build: fn String_lastIndexOf_2(s: *String, needle: *String, startIndex: i32) -> i32
/// MIR wrapper for String.lastIndexOf with explicit startIndex
/// This is used when user writes: s.lastIndexOf(needle, startIndex)
fn build_string_lastindexof_2_wrapper(builder: &mut MirBuilder) {
    let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("String_lastIndexOf_2")
        .param("s", string_ptr_ty.clone())
        .param("needle", string_ptr_ty.clone())
        .param("start_index", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s = builder.get_param(0);
    let needle = builder.get_param(1);
    let start_index = builder.get_param(2);

    // Forward to the runtime function
    let extern_id = builder
        .get_function_by_name("haxe_string_last_index_of_ptr")
        .expect("haxe_string_last_index_of_ptr not found");
    let result = builder
        .call(extern_id, vec![s, needle, start_index])
        .unwrap();

    builder.ret(Some(result));
}

/// Build: fn String_charAt(s: *String, index: i32) -> *String
/// MIR wrapper for String.charAt that forwards to the runtime function
fn build_string_charat_wrapper(builder: &mut MirBuilder) {
    let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("String_charAt")
        .param("s", string_ptr_ty.clone())
        .param("index", i32_ty.clone())
        .returns(string_ptr_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s = builder.get_param(0);
    let index = builder.get_param(1);

    // Forward to the runtime function
    let extern_id = builder
        .get_function_by_name("haxe_string_char_at_ptr")
        .expect("haxe_string_char_at_ptr not found");
    let result = builder.call(extern_id, vec![s, index]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn String_substring(s: *String, startIndex: i32, endIndex: i32) -> *String
/// MIR wrapper for String.substring that forwards to the runtime function
fn build_string_substring_wrapper(builder: &mut MirBuilder) {
    let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("String_substring")
        .param("s", string_ptr_ty.clone())
        .param("start_index", i32_ty.clone())
        .param("end_index", i32_ty.clone())
        .returns(string_ptr_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let s = builder.get_param(0);
    let start_index = builder.get_param(1);
    let end_index = builder.get_param(2);

    // Forward to the runtime function
    let extern_id = builder
        .get_function_by_name("haxe_string_substring_ptr")
        .expect("haxe_string_substring_ptr not found");
    let result = builder
        .call(extern_id, vec![s, start_index, end_index])
        .unwrap();

    builder.ret(Some(result));
}
