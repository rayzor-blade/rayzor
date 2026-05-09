//! `Select` MIR wrappers — multi-channel non-deterministic receive.
//!
//! The user calls `Select.recv(channels:Array<Dynamic>)` /
//! `Select.tryRecv(...)`. The MIR wrapper extracts the underlying contiguous
//! channel-handle buffer + length from the HaxeArray and forwards to the
//! runtime function `rayzor_select_recv` / `rayzor_select_try_recv`, which
//! returns a heap-allocated `SelectResult` ({ index: i64, value: *u8 }).
//!
//! HaxeArray layout (matches the `Tensor_zeros` etc. wrappers):
//! - offset 0:  data pointer (i64 / i32 on WASM)
//! - offset 8:  length        (same)
//! - offset 16: capacity
//! - offset 24: elem_size

use crate::ir::mir_builder::MirBuilder;
use crate::ir::{BinaryOp, CallingConvention, IrType};

pub fn build_select_type(builder: &mut MirBuilder) {
    declare_select_externs(builder);
    build_select_recv(builder);
    build_select_try_recv(builder);
}

fn declare_select_externs(builder: &mut MirBuilder) {
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
    let i32_ty = IrType::I32;

    // rayzor_select_recv(channels_ptr: *u8, count: i32) -> *SelectResult
    let func_id = builder
        .begin_function("rayzor_select_recv")
        .param("channels_ptr", ptr_u8.clone())
        .param("count", i32_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_select_try_recv(channels_ptr: *u8, count: i32) -> *SelectResult
    let func_id = builder
        .begin_function("rayzor_select_try_recv")
        .param("channels_ptr", ptr_u8.clone())
        .param("count", i32_ty)
        .returns(ptr_u8)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Extract `(data_ptr, length)` from a HaxeArray pointer at offsets 0 / 8.
fn extract_array_ptr_len(
    builder: &mut MirBuilder,
    arr: crate::ir::IrId,
) -> (crate::ir::IrId, crate::ir::IrId) {
    let i64_ty = IrType::I64;
    let data_ptr = builder.load(arr, i64_ty.clone());
    let eight = builder.const_i64(8);
    let len_addr = builder.bin_op(BinaryOp::Add, arr, eight);
    let len = builder.load(len_addr, i64_ty);
    (data_ptr, len)
}

/// Select_recv(channels_arr: *u8) -> *SelectResult
fn build_select_recv(builder: &mut MirBuilder) {
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("Select_recv")
        .param("channels_arr", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channels_arr = builder.get_param(0);
    let (data_ptr, count) = extract_array_ptr_len(builder, channels_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_select_recv")
        .expect("rayzor_select_recv not found");
    let result = builder.call(extern_id, vec![data_ptr, count]).unwrap();
    builder.ret(Some(result));
}

/// Select_tryRecv(channels_arr: *u8) -> *SelectResult
fn build_select_try_recv(builder: &mut MirBuilder) {
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

    let func_id = builder
        .begin_function("Select_tryRecv")
        .param("channels_arr", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channels_arr = builder.get_param(0);
    let (data_ptr, count) = extract_array_ptr_len(builder, channels_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_select_try_recv")
        .expect("rayzor_select_try_recv not found");
    let result = builder.call(extern_id, vec![data_ptr, count]).unwrap();
    builder.ret(Some(result));
}
