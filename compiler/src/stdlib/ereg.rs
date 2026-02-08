/// EReg (Regular Expression) type implementation using MIR Builder
///
/// EReg is an opaque pointer type backed by Rust's `regex` crate.
/// All methods are extern C calls to runtime functions.
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// Build all EReg type functions
pub fn build_ereg_type(builder: &mut MirBuilder) {
    declare_ereg_externs(builder);
    build_ereg_match_sub_2(builder);
}

/// Declare EReg extern runtime functions
fn declare_ereg_externs(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
    let string_ty = IrType::String;
    let i32_ty = IrType::I32;
    let void_ty = IrType::Void;

    // haxe_ereg_new(pattern: *const HaxeString, opts: *const HaxeString) -> *mut u8
    let func_id = builder
        .begin_function("haxe_ereg_new")
        .param("pattern", string_ty.clone())
        .param("opts", string_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_ereg_match(ereg: *mut u8, s: *const HaxeString) -> i32
    let func_id = builder
        .begin_function("haxe_ereg_match")
        .param("ereg", ptr_u8.clone())
        .param("s", string_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_ereg_matched(ereg: *mut u8, n: i32) -> String
    let func_id = builder
        .begin_function("haxe_ereg_matched")
        .param("ereg", ptr_u8.clone())
        .param("n", i32_ty.clone())
        .returns(string_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_ereg_matched_left(ereg: *mut u8) -> String
    let func_id = builder
        .begin_function("haxe_ereg_matched_left")
        .param("ereg", ptr_u8.clone())
        .returns(string_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_ereg_matched_right(ereg: *mut u8) -> String
    let func_id = builder
        .begin_function("haxe_ereg_matched_right")
        .param("ereg", ptr_u8.clone())
        .returns(string_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_ereg_matched_pos(ereg: *mut u8, out_pos: *mut i32, out_len: *mut i32)
    let func_id = builder
        .begin_function("haxe_ereg_matched_pos")
        .param("ereg", ptr_u8.clone())
        .param("out_pos", ptr_void.clone())
        .param("out_len", ptr_void.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_ereg_match_sub(ereg: *mut u8, s: *const HaxeString, pos: i32, len: i32) -> i32
    let func_id = builder
        .begin_function("haxe_ereg_match_sub")
        .param("ereg", ptr_u8.clone())
        .param("s", string_ty.clone())
        .param("pos", i32_ty.clone())
        .param("len", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_ereg_split(ereg: *mut u8, s: *const HaxeString) -> *mut HaxeArray
    let func_id = builder
        .begin_function("haxe_ereg_split")
        .param("ereg", ptr_u8.clone())
        .param("s", string_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_ereg_replace(ereg: *mut u8, s: *const HaxeString, by: *const HaxeString) -> String
    let func_id = builder
        .begin_function("haxe_ereg_replace")
        .param("ereg", ptr_u8.clone())
        .param("s", string_ty.clone())
        .param("by", string_ty.clone())
        .returns(string_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // haxe_ereg_escape(s: *const HaxeString) -> String
    let func_id = builder
        .begin_function("haxe_ereg_escape")
        .param("s", string_ty.clone())
        .returns(string_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// MIR wrapper for matchSub with 2 params (default len = -1)
/// EReg_matchSub_2(ereg, s, pos) → haxe_ereg_match_sub(ereg, s, pos, -1)
fn build_ereg_match_sub_2(builder: &mut MirBuilder) {
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
    let string_ty = IrType::String;
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("EReg_matchSub_2")
        .param("ereg", ptr_u8)
        .param("s", string_ty)
        .param("pos", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let ereg = builder.get_param(0);
    let s = builder.get_param(1);
    let pos = builder.get_param(2);
    let default_len = builder.const_i32(-1);

    let extern_id = builder
        .get_function_by_name("haxe_ereg_match_sub")
        .expect("haxe_ereg_match_sub extern not found");
    let result = builder
        .call(extern_id, vec![ereg, s, pos, default_len])
        .unwrap();
    builder.ret(Some(result));
}
