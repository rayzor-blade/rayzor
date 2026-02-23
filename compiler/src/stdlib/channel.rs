/// Channel: Multi-producer, multi-consumer message passing
///
/// This module provides MIR implementations for channel operations.
/// The actual channel implementation is delegated to extern runtime functions.
///
/// Memory layout:
/// ```ignore
/// struct Channel<T> {
///     inner: *u8,     // Opaque channel handle
///     capacity: i32,  // Buffer capacity (0 = unbounded)
/// }
/// ```
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// Build all Channel functions
pub fn build_channel_type(builder: &mut MirBuilder) {
    // Declare extern runtime functions first
    declare_channel_externs(builder);

    // Build wrapper functions
    build_channel_init(builder);
    build_channel_send(builder);
    build_channel_try_send(builder);
    build_channel_receive(builder);
    build_channel_try_receive(builder);
    build_channel_close(builder);
    build_channel_is_closed(builder);
    build_channel_len(builder);
    build_channel_capacity(builder);
    build_channel_is_empty(builder);
    build_channel_is_full(builder);
}

/// Declare extern runtime functions
fn declare_channel_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // extern fn rayzor_channel_init(capacity: i32) -> *u8
    let func_id = builder
        .begin_function("rayzor_channel_init")
        .param("capacity", i32_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_channel_send(channel: *u8, value: *u8)
    let func_id = builder
        .begin_function("rayzor_channel_send")
        .param("channel", ptr_u8.clone())
        .param("value", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_channel_try_send(channel: *u8, value: *u8) -> bool
    let func_id = builder
        .begin_function("rayzor_channel_try_send")
        .param("channel", ptr_u8.clone())
        .param("value", ptr_u8.clone())
        .returns(bool_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_channel_receive(channel: *u8) -> *u8
    let func_id = builder
        .begin_function("rayzor_channel_receive")
        .param("channel", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_channel_try_receive(channel: *u8) -> *u8 (null if empty)
    let func_id = builder
        .begin_function("rayzor_channel_try_receive")
        .param("channel", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_channel_close(channel: *u8)
    let func_id = builder
        .begin_function("rayzor_channel_close")
        .param("channel", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_channel_is_closed(channel: *u8) -> bool
    let func_id = builder
        .begin_function("rayzor_channel_is_closed")
        .param("channel", ptr_u8.clone())
        .returns(bool_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_channel_len(channel: *u8) -> i32
    let func_id = builder
        .begin_function("rayzor_channel_len")
        .param("channel", ptr_u8.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_channel_capacity(channel: *u8) -> i32
    let func_id = builder
        .begin_function("rayzor_channel_capacity")
        .param("channel", ptr_u8.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_channel_is_empty(channel: *u8) -> bool
    let func_id = builder
        .begin_function("rayzor_channel_is_empty")
        .param("channel", ptr_u8.clone())
        .returns(bool_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_channel_is_full(channel: *u8) -> bool
    let func_id = builder
        .begin_function("rayzor_channel_is_full")
        .param("channel", ptr_u8.clone())
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Build: fn Channel_init(capacity: i32) -> *Channel
fn build_channel_init(builder: &mut MirBuilder) {
    let i32_ty = builder.i32_type();
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Channel_init")
        .param("capacity", i32_ty)
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let capacity = builder.get_param(0);

    let new_id = builder
        .get_function_by_name("rayzor_channel_init")
        .expect("rayzor_channel_init not found");
    let handle = builder.call(new_id, vec![capacity]).unwrap();

    builder.ret(Some(handle));
}

/// Build: fn Channel_send(channel: *Channel, value: *u8)
fn build_channel_send(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("Channel_send")
        .param("channel", ptr_u8.clone())
        .param("value", ptr_u8)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channel = builder.get_param(0);
    let value = builder.get_param(1);

    let send_id = builder
        .get_function_by_name("rayzor_channel_send")
        .expect("rayzor_channel_send not found");
    let _result = builder.call(send_id, vec![channel, value]);

    builder.ret(None);
}

/// Build: fn Channel_trySend(channel: *Channel, value: *u8) -> bool
fn build_channel_try_send(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Channel_trySend")
        .param("channel", ptr_u8.clone())
        .param("value", ptr_u8)
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channel = builder.get_param(0);
    let value = builder.get_param(1);

    let try_send_id = builder
        .get_function_by_name("rayzor_channel_try_send")
        .expect("rayzor_channel_try_send not found");
    let result = builder.call(try_send_id, vec![channel, value]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Channel_receive(channel: *Channel) -> *u8
fn build_channel_receive(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Channel_receive")
        .param("channel", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channel = builder.get_param(0);

    let receive_id = builder
        .get_function_by_name("rayzor_channel_receive")
        .expect("rayzor_channel_receive not found");
    let result = builder.call(receive_id, vec![channel]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Channel_tryReceive(channel: *Channel) -> *u8
fn build_channel_try_receive(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Channel_tryReceive")
        .param("channel", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channel = builder.get_param(0);

    let try_receive_id = builder
        .get_function_by_name("rayzor_channel_try_receive")
        .expect("rayzor_channel_try_receive not found");
    let result = builder.call(try_receive_id, vec![channel]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Channel_close(channel: *Channel)
fn build_channel_close(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("Channel_close")
        .param("channel", ptr_u8)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channel = builder.get_param(0);

    let close_id = builder
        .get_function_by_name("rayzor_channel_close")
        .expect("rayzor_channel_close not found");
    let _result = builder.call(close_id, vec![channel]);

    builder.ret(None);
}

/// Build: fn Channel_isClosed(channel: *Channel) -> bool
fn build_channel_is_closed(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Channel_isClosed")
        .param("channel", ptr_u8)
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channel = builder.get_param(0);

    let is_closed_id = builder
        .get_function_by_name("rayzor_channel_is_closed")
        .expect("rayzor_channel_is_closed not found");
    let result = builder.call(is_closed_id, vec![channel]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Channel_len(channel: *Channel) -> i32
fn build_channel_len(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();

    let func_id = builder
        .begin_function("Channel_len")
        .param("channel", ptr_u8)
        .returns(i32_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channel = builder.get_param(0);

    let len_id = builder
        .get_function_by_name("rayzor_channel_len")
        .expect("rayzor_channel_len not found");
    let result = builder.call(len_id, vec![channel]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Channel_capacity(channel: *Channel) -> i32
fn build_channel_capacity(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();

    let func_id = builder
        .begin_function("Channel_capacity")
        .param("channel", ptr_u8)
        .returns(i32_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channel = builder.get_param(0);

    let capacity_id = builder
        .get_function_by_name("rayzor_channel_capacity")
        .expect("rayzor_channel_capacity not found");
    let result = builder.call(capacity_id, vec![channel]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Channel_isEmpty(channel: *Channel) -> bool
fn build_channel_is_empty(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Channel_isEmpty")
        .param("channel", ptr_u8)
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channel = builder.get_param(0);

    let is_empty_id = builder
        .get_function_by_name("rayzor_channel_is_empty")
        .expect("rayzor_channel_is_empty not found");
    let result = builder.call(is_empty_id, vec![channel]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Channel_isFull(channel: *Channel) -> bool
fn build_channel_is_full(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Channel_isFull")
        .param("channel", ptr_u8)
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let channel = builder.get_param(0);

    let is_full_id = builder
        .get_function_by_name("rayzor_channel_is_full")
        .expect("rayzor_channel_is_full not found");
    let result = builder.call(is_full_id, vec![channel]).unwrap();

    builder.ret(Some(result));
}
