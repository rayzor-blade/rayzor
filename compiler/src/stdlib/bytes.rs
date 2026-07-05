//! MIR helpers for hot `rayzor.Bytes` access patterns.
//!
//! The public `getInt32` / `setInt32` runtime functions intentionally use
//! unaligned loads because file formats can read at arbitrary byte offsets.
//! Kernel-owned scratch buffers use aligned cells in tight loops, so expose
//! explicit unchecked aligned helpers that lower to direct GEP/load/store.

use crate::ir::mir_builder::MirBuilder;
use crate::ir::{BinaryOp, CallingConvention, InlineHint, IrType};

const HAXE_BYTES_DATA_OFFSET: i64 = 0;

pub fn build_bytes_type(builder: &mut MirBuilder) {
    build_load_i32_aligned_unchecked(builder);
    build_store_i32_aligned_unchecked(builder);
    build_mem_accessors(builder);
}

/// Address-based unchecked loads/stores (`rayzor.Mem`). The caller holds a
/// raw base address (`Bytes.address()`, `Tensor.data().raw()`, `QTensor.
/// dataPtr()`), so these skip even the handle deref — one typed load/store.
/// The f32 pair is the only Haxe-reachable scalar f32 access (Haxe `Float`
/// is f64; `Ptr<Float>` reads/writes 8 bytes and corrupts F32 buffers).
fn build_mem_accessors(builder: &mut MirBuilder) {
    build_mem_load(builder, "rayzor_mem_load_f32", IrType::F32, IrType::F64);
    build_mem_store(builder, "rayzor_mem_store_f32", IrType::F64, IrType::F32);
    build_mem_load(builder, "rayzor_mem_load_f64", IrType::F64, IrType::F64);
    build_mem_store(builder, "rayzor_mem_store_f64", IrType::F64, IrType::F64);
    build_mem_load(builder, "rayzor_mem_load_u8", IrType::U8, IrType::I32);
    build_mem_store(builder, "rayzor_mem_store_u8", IrType::I32, IrType::U8);
    build_mem_load(builder, "rayzor_mem_load_i32", IrType::I32, IrType::I32);
    build_mem_store(builder, "rayzor_mem_store_i32", IrType::I32, IrType::I32);
}

/// `name(addr: i64) -> ret_ty` — load `mem_ty` at addr, widen to `ret_ty`.
fn build_mem_load(builder: &mut MirBuilder, name: &str, mem_ty: IrType, ret_ty: IrType) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function(name)
        .param("addr", i64_ty)
        .returns(ret_ty.clone())
        .calling_convention(CallingConvention::C)
        .inline(InlineHint::Always)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let addr = builder.get_param(0);
    let raw = builder.load(addr, mem_ty.clone());
    let value = if mem_ty == ret_ty {
        raw
    } else {
        builder.cast(raw, mem_ty, ret_ty)
    };
    builder.ret(Some(value));
}

/// `name(addr: i64, value: arg_ty)` — narrow to `mem_ty`, store at addr.
fn build_mem_store(builder: &mut MirBuilder, name: &str, arg_ty: IrType, mem_ty: IrType) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function(name)
        .param("addr", i64_ty)
        .param("value", arg_ty.clone())
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .inline(InlineHint::Always)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let addr = builder.get_param(0);
    let value = builder.get_param(1);
    let narrowed = if arg_ty == mem_ty {
        value
    } else {
        builder.cast(value, arg_ty, mem_ty)
    };
    builder.store(addr, narrowed);
    builder.ret(None);
}

fn build_load_i32_aligned_unchecked(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let i32_ty = IrType::I32;
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("rayzor_bytes_load_i32_aligned_unchecked")
        .param("bytes", ptr_void)
        .param("pos", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .inline(InlineHint::Always)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let bytes = builder.get_param(0);
    let pos = builder.get_param(1);
    let data_ptr = bytes_data_ptr(builder, bytes);
    let pos64 = builder.cast(pos, i32_ty, i64_ty.clone());
    let elem_ptr = builder.bin_op(BinaryOp::Add, data_ptr, pos64);
    let value = builder.load(elem_ptr, IrType::I32);
    builder.ret(Some(value));
}

fn build_store_i32_aligned_unchecked(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let i32_ty = IrType::I32;
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("rayzor_bytes_store_i32_aligned_unchecked")
        .param("bytes", ptr_void)
        .param("pos", i32_ty.clone())
        .param("value", i32_ty.clone())
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .inline(InlineHint::Always)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let bytes = builder.get_param(0);
    let pos = builder.get_param(1);
    let value = builder.get_param(2);
    let data_ptr = bytes_data_ptr(builder, bytes);
    let pos64 = builder.cast(pos, i32_ty, i64_ty.clone());
    let elem_ptr = builder.bin_op(BinaryOp::Add, data_ptr, pos64);
    builder.store(elem_ptr, value);
    builder.ret(None);
}

fn bytes_data_ptr(builder: &mut MirBuilder, bytes: crate::ir::IrId) -> crate::ir::IrId {
    if HAXE_BYTES_DATA_OFFSET == 0 {
        builder.load(bytes, IrType::I64)
    } else {
        let off = builder.const_i64(HAXE_BYTES_DATA_OFFSET);
        let field = builder.bin_op(BinaryOp::Add, bytes, off);
        builder.load(field, IrType::I64)
    }
}
