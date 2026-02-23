/// Vec<u8>: Performance-optimized concrete vector for bytes
///
/// This is a complete MIR implementation of a dynamic byte vector,
/// used as the backing storage for strings and byte buffers.
///
/// Memory layout:
/// ```ignore
/// struct Vec<u8> {
///     ptr: *u8,      // Pointer to heap-allocated array
///     len: u64,      // Number of elements
///     cap: u64,      // Allocated capacity
/// }
/// ```
///
/// Growth strategy:
/// - Initial capacity: 16 bytes
/// - Growth factor: 2x (16 → 32 → 64 → 128...)
/// - Uses C realloc() for resizing
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{BinaryOp, CompareOp, IrType};

/// Build all Vec<u8> functions
pub fn build_vec_u8_type(builder: &mut MirBuilder) {
    build_vec_u8_new(builder);
    build_vec_u8_push(builder);
    build_vec_u8_pop(builder);
    build_vec_u8_get(builder);
    build_vec_u8_set(builder);
    build_vec_u8_len(builder);
    build_vec_u8_capacity(builder);
    build_vec_u8_clear(builder);
    build_vec_u8_free(builder);
}

/// Get the Vec<u8> struct type
fn get_vec_u8_type(builder: &MirBuilder) -> IrType {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let usize_ty = builder.u64_type();

    builder.struct_type(Some("vec_u8"), vec![ptr_u8_ty, usize_ty.clone(), usize_ty])
}

/// Build: fn vec_u8_new() -> Vec<u8>
/// Creates an empty vector with initial capacity of 16 bytes
fn build_vec_u8_new(builder: &mut MirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let usize_ty = builder.u64_type();
    let vec_u8_ty = get_vec_u8_type(builder);

    let func_id = builder
        .begin_function("vec_u8_new")
        .returns(vec_u8_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Allocate initial capacity of 16 bytes
    let initial_cap = builder.const_u64(16);
    let u8_size = builder.const_u64(1);
    let alloc_size = builder.mul(initial_cap, u8_size, usize_ty.clone());

    // Call malloc to allocate memory
    let malloc_id = builder
        .get_function_by_name("malloc")
        .expect("malloc not found");
    let malloc_ref = builder.function_ref(malloc_id);
    let ptr_u8 = builder.call(malloc_id, vec![alloc_size]).unwrap();

    // Create Vec<u8> struct: { ptr, len: 0, cap: 16 }
    let zero = builder.const_u64(0);
    let vec_value = builder.create_struct(vec_u8_ty, vec![ptr_u8, zero, initial_cap]);

    builder.ret(Some(vec_value));
}

/// Build: fn vec_u8_push(vec: *Vec<u8>, value: u8)
/// Appends an element, growing the vector if necessary
fn build_vec_u8_push(builder: &mut MirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();
    let bool_ty = builder.bool_type();
    let vec_u8_ty = get_vec_u8_type(builder);
    let ptr_vec_ty = builder.ptr_type(vec_u8_ty.clone());

    let func_id = builder
        .begin_function("vec_u8_push")
        .param("vec", ptr_vec_ty.clone())
        .param("value", u8_ty.clone())
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let check_capacity = builder.create_block("check_capacity");
    let need_grow = builder.create_block("need_grow");
    let no_grow = builder.create_block("no_grow");
    let insert_element = builder.create_block("insert_element");

    // Entry: load vec and extract fields
    builder.set_insert_point(entry);
    let vec_ptr = builder.get_param(0);
    let value = builder.get_param(1);

    let vec_val = builder.load(vec_ptr, vec_u8_ty.clone());
    let ptr_field = builder.extract_field(vec_val, 0);
    let len_field = builder.extract_field(vec_val, 1);
    let cap_field = builder.extract_field(vec_val, 2);

    builder.br(check_capacity);

    // Check if len == cap (need to grow)
    builder.set_insert_point(check_capacity);
    let is_full = builder.icmp(CompareOp::Eq, len_field, cap_field, bool_ty);
    builder.cond_br(is_full, need_grow, no_grow);

    // Grow: new_cap = cap * 2, realloc
    builder.set_insert_point(need_grow);
    let two = builder.const_u64(2);
    let new_cap = builder.mul(cap_field, two, usize_ty.clone());
    let u8_size = builder.const_u64(1);
    let old_size = builder.mul(cap_field, u8_size, usize_ty.clone());
    let new_size = builder.mul(new_cap, u8_size, usize_ty.clone());

    // Call realloc (libc signature: ptr, new_size only)
    let realloc_id = builder
        .get_function_by_name("realloc")
        .expect("realloc not found");
    let new_ptr_u8 = builder.call(realloc_id, vec![ptr_field, new_size]).unwrap();

    // Create updated vec with new pointer and capacity
    let grown_vec = builder.create_struct(vec_u8_ty.clone(), vec![new_ptr_u8, len_field, new_cap]);
    builder.store(vec_ptr, grown_vec);
    builder.br(insert_element);

    // No grow needed
    builder.set_insert_point(no_grow);
    builder.br(insert_element);

    // Insert element at vec[len]
    builder.set_insert_point(insert_element);

    // Reload vec (may have been updated in grow path)
    let vec_val2 = builder.load(vec_ptr, vec_u8_ty.clone());
    let ptr_field2 = builder.extract_field(vec_val2, 0);
    let len_field2 = builder.extract_field(vec_val2, 1);
    let cap_field2 = builder.extract_field(vec_val2, 2);

    // Calculate element pointer: ptr + len
    let elem_ptr = builder.ptr_add(ptr_field2, len_field2, ptr_u8_ty.clone());
    builder.store(elem_ptr, value);

    // Increment length
    let one = builder.const_u64(1);
    let new_len = builder.add(len_field2, one, usize_ty.clone());

    // Store updated vec
    let final_vec = builder.create_struct(vec_u8_ty, vec![ptr_field2, new_len, cap_field2]);
    builder.store(vec_ptr, final_vec);

    // Void function - return nothing
    builder.ret(None);
}

/// Build: fn vec_u8_pop(vec: *Vec<u8>) -> Option<u8>
/// Removes and returns the last element, or None if empty
fn build_vec_u8_pop(builder: &mut MirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();
    let bool_ty = builder.bool_type();
    let vec_u8_ty = get_vec_u8_type(builder);
    let ptr_vec_ty = builder.ptr_type(vec_u8_ty.clone());

    // Create Option<u8> type
    let option_variants = vec![
        crate::ir::UnionVariant {
            name: "None".to_string(),
            tag: 0,
            fields: vec![void_ty.clone()],
        },
        crate::ir::UnionVariant {
            name: "Some".to_string(),
            tag: 1,
            fields: vec![u8_ty.clone()],
        },
    ];
    let option_ty = builder.union_type(Some("Option"), option_variants);

    let func_id = builder
        .begin_function("vec_u8_pop")
        .param("vec", ptr_vec_ty)
        .returns(option_ty.clone())
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let is_empty_check = builder.create_block("is_empty_check");
    let empty_case = builder.create_block("empty_case");
    let non_empty_case = builder.create_block("non_empty_case");

    // Entry
    builder.set_insert_point(entry);
    let vec_ptr = builder.get_param(0);

    let vec_val = builder.load(vec_ptr, vec_u8_ty.clone());
    let ptr_field = builder.extract_field(vec_val, 0);
    let len_field = builder.extract_field(vec_val, 1);
    let cap_field = builder.extract_field(vec_val, 2);

    builder.br(is_empty_check);

    // Check if len == 0
    builder.set_insert_point(is_empty_check);
    let zero = builder.const_u64(0);
    let is_empty = builder.icmp(CompareOp::Eq, len_field, zero, bool_ty);
    builder.cond_br(is_empty, empty_case, non_empty_case);

    // Empty: return None
    builder.set_insert_point(empty_case);
    let undef_val = builder.undef(void_ty);
    let none_val = builder.create_union(0, undef_val, option_ty.clone());
    builder.ret(Some(none_val));

    // Non-empty: return Some(last element)
    builder.set_insert_point(non_empty_case);

    // Decrement length
    let one = builder.const_u64(1);
    let new_len = builder.sub(len_field, one, usize_ty.clone());

    // Load element at vec[new_len]
    let elem_ptr = builder.ptr_add(ptr_field, new_len, ptr_u8_ty.clone());
    let elem_val = builder.load(elem_ptr, u8_ty);

    // Update vec with new length
    let updated_vec = builder.create_struct(vec_u8_ty, vec![ptr_field, new_len, cap_field]);
    builder.store(vec_ptr, updated_vec);

    // Return Some(elem_val)
    let some_val = builder.create_union(1, elem_val, option_ty);
    builder.ret(Some(some_val));
}

/// Build: fn vec_u8_get(vec: *Vec<u8>, index: u64) -> Option<u8>
/// Returns element at index, or None if out of bounds
fn build_vec_u8_get(builder: &mut MirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();
    let bool_ty = builder.bool_type();
    let vec_u8_ty = get_vec_u8_type(builder);
    let ptr_vec_ty = builder.ptr_type(vec_u8_ty.clone());

    // Create Option<u8> type
    let option_variants = vec![
        crate::ir::UnionVariant {
            name: "None".to_string(),
            tag: 0,
            fields: vec![void_ty.clone()],
        },
        crate::ir::UnionVariant {
            name: "Some".to_string(),
            tag: 1,
            fields: vec![u8_ty.clone()],
        },
    ];
    let option_ty = builder.union_type(Some("Option"), option_variants);

    let func_id = builder
        .begin_function("vec_u8_get")
        .param("vec", ptr_vec_ty)
        .param("index", usize_ty.clone())
        .returns(option_ty.clone())
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let bounds_check = builder.create_block("bounds_check");
    let out_of_bounds = builder.create_block("out_of_bounds");
    let in_bounds = builder.create_block("in_bounds");

    // Entry
    builder.set_insert_point(entry);
    let vec_ptr = builder.get_param(0);
    let index = builder.get_param(1);

    let vec_val = builder.load(vec_ptr, vec_u8_ty);
    let ptr_field = builder.extract_field(vec_val, 0);
    let len_field = builder.extract_field(vec_val, 1);

    builder.br(bounds_check);

    // Check if index >= len
    builder.set_insert_point(bounds_check);
    let is_out_of_bounds = builder.icmp(CompareOp::UGe, index, len_field, bool_ty);
    builder.cond_br(is_out_of_bounds, out_of_bounds, in_bounds);

    // Out of bounds: return None
    builder.set_insert_point(out_of_bounds);
    let undef_val = builder.undef(void_ty);
    let none_val = builder.create_union(0, undef_val, option_ty.clone());
    builder.ret(Some(none_val));

    // In bounds: return Some(vec[index])
    builder.set_insert_point(in_bounds);
    let elem_ptr = builder.ptr_add(ptr_field, index, ptr_u8_ty);
    let elem_val = builder.load(elem_ptr, u8_ty);
    let some_val = builder.create_union(1, elem_val, option_ty);
    builder.ret(Some(some_val));
}

/// Build: fn vec_u8_set(vec: *Vec<u8>, index: u64, value: u8) -> bool
/// Sets element at index, returns false if out of bounds
fn build_vec_u8_set(builder: &mut MirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty.clone());
    let usize_ty = builder.u64_type();
    let bool_ty = builder.bool_type();
    let vec_u8_ty = get_vec_u8_type(builder);
    let ptr_vec_ty = builder.ptr_type(vec_u8_ty.clone());

    let func_id = builder
        .begin_function("vec_u8_set")
        .param("vec", ptr_vec_ty)
        .param("index", usize_ty.clone())
        .param("value", u8_ty)
        .returns(bool_ty.clone())
        .build();

    builder.set_current_function(func_id);

    // Create blocks
    let entry = builder.create_block("entry");
    let bounds_check = builder.create_block("bounds_check");
    let out_of_bounds = builder.create_block("out_of_bounds");
    let in_bounds = builder.create_block("in_bounds");

    // Entry
    builder.set_insert_point(entry);
    let vec_ptr = builder.get_param(0);
    let index = builder.get_param(1);
    let value = builder.get_param(2);

    let vec_val = builder.load(vec_ptr, vec_u8_ty);
    let ptr_field = builder.extract_field(vec_val, 0);
    let len_field = builder.extract_field(vec_val, 1);

    builder.br(bounds_check);

    // Check if index >= len
    builder.set_insert_point(bounds_check);
    let is_out_of_bounds = builder.icmp(CompareOp::UGe, index, len_field, bool_ty.clone());
    builder.cond_br(is_out_of_bounds, out_of_bounds, in_bounds);

    // Out of bounds: return false
    builder.set_insert_point(out_of_bounds);
    let false_val = builder.const_bool(false);
    builder.ret(Some(false_val));

    // In bounds: set vec[index] = value, return true
    builder.set_insert_point(in_bounds);
    let elem_ptr = builder.ptr_add(ptr_field, index, ptr_u8_ty);
    builder.store(elem_ptr, value);
    let true_val = builder.const_bool(true);
    builder.ret(Some(true_val));
}

/// Build: fn vec_u8_len(vec: *Vec<u8>) -> u64
/// Returns the number of elements in the vector
fn build_vec_u8_len(builder: &mut MirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let usize_ty = builder.u64_type();
    let vec_u8_ty = get_vec_u8_type(builder);
    let ptr_vec_ty = builder.ptr_type(vec_u8_ty.clone());

    let func_id = builder
        .begin_function("vec_u8_len")
        .param("vec", ptr_vec_ty)
        .returns(usize_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_ptr = builder.get_param(0);
    let vec_val = builder.load(vec_ptr, vec_u8_ty);
    let len_field = builder.extract_field(vec_val, 1);
    builder.ret(Some(len_field));
}

/// Build: fn vec_u8_capacity(vec: *Vec<u8>) -> u64
/// Returns the allocated capacity
fn build_vec_u8_capacity(builder: &mut MirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let usize_ty = builder.u64_type();
    let vec_u8_ty = get_vec_u8_type(builder);
    let ptr_vec_ty = builder.ptr_type(vec_u8_ty.clone());

    let func_id = builder
        .begin_function("vec_u8_capacity")
        .param("vec", ptr_vec_ty)
        .returns(usize_ty.clone())
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_ptr = builder.get_param(0);
    let vec_val = builder.load(vec_ptr, vec_u8_ty);
    let cap_field = builder.extract_field(vec_val, 2);
    builder.ret(Some(cap_field));
}

/// Build: fn vec_u8_clear(vec: *Vec<u8>)
/// Resets length to 0 (keeps capacity)
fn build_vec_u8_clear(builder: &mut MirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();
    let vec_u8_ty = get_vec_u8_type(builder);
    let ptr_vec_ty = builder.ptr_type(vec_u8_ty.clone());

    let func_id = builder
        .begin_function("vec_u8_clear")
        .param("vec", ptr_vec_ty)
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_ptr = builder.get_param(0);
    let vec_val = builder.load(vec_ptr, vec_u8_ty.clone());
    let ptr_field = builder.extract_field(vec_val, 0);
    let cap_field = builder.extract_field(vec_val, 2);

    // Set length to 0
    let zero = builder.const_u64(0);
    let cleared_vec = builder.create_struct(vec_u8_ty, vec![ptr_field, zero, cap_field]);
    builder.store(vec_ptr, cleared_vec);

    // Void function - return nothing
    builder.ret(None);
}

/// Build: fn vec_u8_free(vec: Vec<u8>)
/// Frees the allocated memory
fn build_vec_u8_free(builder: &mut MirBuilder) {
    let u8_ty = builder.u8_type();
    let ptr_u8_ty = builder.ptr_type(u8_ty);
    let usize_ty = builder.u64_type();
    let void_ty = builder.void_type();
    let vec_u8_ty = get_vec_u8_type(builder);

    let func_id = builder
        .begin_function("vec_u8_free")
        .param("vec", vec_u8_ty.clone())
        .returns(void_ty)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let vec_val = builder.get_param(0);
    let ptr_field = builder.extract_field(vec_val, 0);
    let cap_field = builder.extract_field(vec_val, 2);

    // Calculate size from capacity
    let u8_size = builder.const_u64(1);
    let total_size = builder.mul(cap_field, u8_size, usize_ty);

    // Call free (libc signature: ptr only)
    let free_id = builder
        .get_function_by_name("free")
        .expect("free not found");
    let _result = builder.call(free_id, vec![ptr_field]);

    // Void function - return nothing
    builder.ret(None);
}
