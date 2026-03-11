/// DEFLATE Compression/Decompression type implementation using MIR Builder
///
/// Backs `haxe.zip.Compress` and `haxe.zip.Uncompress` with Rust's `flate2` crate.
/// All methods route to extern C runtime functions.
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// Build all Compress/Uncompress type functions
pub fn build_compress_type(builder: &mut MirBuilder) {
    declare_compress_externs(builder);
}

/// Declare Compress/Uncompress extern runtime functions
fn declare_compress_externs(builder: &mut MirBuilder) {
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
    let i32_ty = IrType::I32;
    let void_ty = IrType::Void;

    // rayzor_compress_new(level: i32) -> *mut u8
    let func_id = builder
        .begin_function("rayzor_compress_new")
        .param("level", i32_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_compress_execute(handle, src, src_pos, dst, dst_pos) -> *mut u8 (anon)
    let func_id = builder
        .begin_function("rayzor_compress_execute")
        .param("handle", ptr_u8.clone())
        .param("src", ptr_u8.clone())
        .param("src_pos", i32_ty.clone())
        .param("dst", ptr_u8.clone())
        .param("dst_pos", i32_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_compress_set_flush(handle, mode: i32)
    let func_id = builder
        .begin_function("rayzor_compress_set_flush")
        .param("handle", ptr_u8.clone())
        .param("mode", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_compress_close(handle)
    let func_id = builder
        .begin_function("rayzor_compress_close")
        .param("handle", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_compress_run(bytes, level) -> *mut u8 (Bytes ptr)
    let func_id = builder
        .begin_function("rayzor_compress_run")
        .param("src", ptr_u8.clone())
        .param("level", i32_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_uncompress_new(window_bits: i32) -> *mut u8
    let func_id = builder
        .begin_function("rayzor_uncompress_new")
        .param("window_bits", i32_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_uncompress_execute(handle, src, src_pos, dst, dst_pos) -> *mut u8 (anon)
    let func_id = builder
        .begin_function("rayzor_uncompress_execute")
        .param("handle", ptr_u8.clone())
        .param("src", ptr_u8.clone())
        .param("src_pos", i32_ty.clone())
        .param("dst", ptr_u8.clone())
        .param("dst_pos", i32_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_uncompress_set_flush(handle, mode: i32)
    let func_id = builder
        .begin_function("rayzor_uncompress_set_flush")
        .param("handle", ptr_u8.clone())
        .param("mode", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_uncompress_close(handle)
    let func_id = builder
        .begin_function("rayzor_uncompress_close")
        .param("handle", ptr_u8.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}
