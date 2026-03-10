/// Networking: Socket and Host MIR wrappers
///
/// Provides MIR implementations for sys.net.Socket and sys.net.Host.
/// The actual networking is delegated to extern runtime functions in
/// runtime/src/socket.rs.
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// Build all Socket and Host functions
pub fn build_socket_type(builder: &mut MirBuilder) {
    declare_socket_externs(builder);
    declare_host_externs(builder);

    // Socket MIR wrappers
    build_socket_new(builder);
    build_socket_connect(builder);
    build_socket_bind(builder);
    build_socket_listen(builder);
    build_socket_accept(builder);
    build_socket_close(builder);
    build_socket_read(builder);
    build_socket_write(builder);
    build_socket_shutdown(builder);
    build_socket_set_blocking(builder);
    build_socket_set_timeout(builder);
    build_socket_set_fast_send(builder);
    build_socket_wait_for_read(builder);

    // Host MIR wrappers
    build_host_new(builder);
    build_host_to_string(builder);
    build_host_reverse(builder);
    build_host_localhost(builder);
}

// =============================================================================
// Extern declarations
// =============================================================================

fn declare_socket_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let void_ty = builder.void_type();
    let f64_ty = IrType::F64;

    // rayzor_socket_new() -> *u8
    let id = builder
        .begin_function("rayzor_socket_new")
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_connect(handle: *u8, host_ip: i32, port: i32)
    let id = builder
        .begin_function("rayzor_socket_connect")
        .param("handle", ptr_u8.clone())
        .param("host_ip", i32_ty.clone())
        .param("port", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_bind(handle: *u8, host_ip: i32, port: i32)
    let id = builder
        .begin_function("rayzor_socket_bind")
        .param("handle", ptr_u8.clone())
        .param("host_ip", i32_ty.clone())
        .param("port", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_listen(handle: *u8, backlog: i32)
    let id = builder
        .begin_function("rayzor_socket_listen")
        .param("handle", ptr_u8.clone())
        .param("backlog", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_accept(handle: *u8) -> *u8
    let id = builder
        .begin_function("rayzor_socket_accept")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_close(handle: *u8)
    let id = builder
        .begin_function("rayzor_socket_close")
        .param("handle", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_read(handle: *u8) -> *u8 (HaxeString*)
    let id = builder
        .begin_function("rayzor_socket_read")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_write(handle: *u8, data: *u8)
    let id = builder
        .begin_function("rayzor_socket_write")
        .param("handle", ptr_u8.clone())
        .param("data", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_shutdown(handle: *u8, read: i32, write: i32)
    let id = builder
        .begin_function("rayzor_socket_shutdown")
        .param("handle", ptr_u8.clone())
        .param("read", i32_ty.clone())
        .param("write", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_set_blocking(handle: *u8, b: i32)
    let id = builder
        .begin_function("rayzor_socket_set_blocking")
        .param("handle", ptr_u8.clone())
        .param("b", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_set_timeout(handle: *u8, seconds: f64)
    let id = builder
        .begin_function("rayzor_socket_set_timeout")
        .param("handle", ptr_u8.clone())
        .param("seconds", f64_ty)
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_set_fast_send(handle: *u8, b: i32)
    let id = builder
        .begin_function("rayzor_socket_set_fast_send")
        .param("handle", ptr_u8.clone())
        .param("b", i32_ty)
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_wait_for_read(handle: *u8)
    let id = builder
        .begin_function("rayzor_socket_wait_for_read")
        .param("handle", ptr_u8.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_select(read: *u8, write: *u8, others: *u8, timeout: f64) -> *u8
    let id = builder
        .begin_function("rayzor_socket_select")
        .param("read", ptr_u8.clone())
        .param("write", ptr_u8.clone())
        .param("others", ptr_u8.clone())
        .param("timeout", IrType::F64)
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_peer(handle: *u8, out_host: *u8, out_port: *u8)
    let void_ty2 = builder.void_type();
    let id = builder
        .begin_function("rayzor_socket_peer")
        .param("handle", ptr_u8.clone())
        .param("out_host", ptr_u8.clone())
        .param("out_port", ptr_u8.clone())
        .returns(void_ty2)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_socket_host_info(handle: *u8, out_host: *u8, out_port: *u8)
    let void_ty3 = builder.void_type();
    let id = builder
        .begin_function("rayzor_socket_host_info")
        .param("handle", ptr_u8.clone())
        .param("out_host", ptr_u8.clone())
        .param("out_port", ptr_u8)
        .returns(void_ty3)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);
}

fn declare_host_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();

    // rayzor_host_new(name: *u8) -> *u8
    let id = builder
        .begin_function("rayzor_host_new")
        .param("name", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_host_get_ip(handle: *u8) -> i32
    let id = builder
        .begin_function("rayzor_host_get_ip")
        .param("handle", ptr_u8.clone())
        .returns(i32_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_host_to_string(handle: *u8) -> *u8
    let id = builder
        .begin_function("rayzor_host_to_string")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_host_reverse(handle: *u8) -> *u8
    let id = builder
        .begin_function("rayzor_host_reverse")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);

    // rayzor_host_localhost() -> *u8
    let id = builder
        .begin_function("rayzor_host_localhost")
        .returns(ptr_u8)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(id);
}

// =============================================================================
// Socket MIR wrapper functions
// =============================================================================

/// Socket_new() -> *u8
fn build_socket_new(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let func_id = builder
        .begin_function("sys_net_Socket_new")
        .returns(ptr_u8)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let rt = builder.get_function_by_name("rayzor_socket_new").unwrap();
    let result = builder.call(rt, vec![]).unwrap();
    builder.ret(Some(result));
}

/// Socket_connect(self: *u8, host: *u8, port: i32)
/// Extracts IP from HostHandle, then calls rayzor_socket_connect.
fn build_socket_connect(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("sys_net_Socket_connect")
        .param("self", ptr_u8.clone())
        .param("host", ptr_u8.clone())
        .param("port", i32_ty)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let host_reg = builder.get_param(1);
    let port_reg = builder.get_param(2);

    // Get the host IP from the HostHandle
    let get_ip = builder.get_function_by_name("rayzor_host_get_ip").unwrap();
    let host_ip = builder.call(get_ip, vec![host_reg]).unwrap();

    let connect = builder
        .get_function_by_name("rayzor_socket_connect")
        .unwrap();
    builder.call(connect, vec![self_reg, host_ip, port_reg]);
    builder.ret(None);
}

/// Socket_bind(self: *u8, host: *u8, port: i32)
fn build_socket_bind(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("sys_net_Socket_bind")
        .param("self", ptr_u8.clone())
        .param("host", ptr_u8.clone())
        .param("port", i32_ty)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let host_reg = builder.get_param(1);
    let port_reg = builder.get_param(2);

    let get_ip = builder.get_function_by_name("rayzor_host_get_ip").unwrap();
    let host_ip = builder.call(get_ip, vec![host_reg]).unwrap();

    let bind = builder.get_function_by_name("rayzor_socket_bind").unwrap();
    builder.call(bind, vec![self_reg, host_ip, port_reg]);
    builder.ret(None);
}

/// Socket_listen(self: *u8, connections: i32)
fn build_socket_listen(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("sys_net_Socket_listen")
        .param("self", ptr_u8.clone())
        .param("connections", i32_ty)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let connections = builder.get_param(1);

    let listen = builder
        .get_function_by_name("rayzor_socket_listen")
        .unwrap();
    builder.call(listen, vec![self_reg, connections]);
    builder.ret(None);
}

/// Socket_accept(self: *u8) -> *u8
fn build_socket_accept(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("sys_net_Socket_accept")
        .param("self", ptr_u8.clone())
        .returns(ptr_u8)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let accept = builder
        .get_function_by_name("rayzor_socket_accept")
        .unwrap();
    let result = builder.call(accept, vec![self_reg]).unwrap();
    builder.ret(Some(result));
}

/// Socket_close(self: *u8)
fn build_socket_close(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("sys_net_Socket_close")
        .param("self", ptr_u8)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let close = builder.get_function_by_name("rayzor_socket_close").unwrap();
    builder.call(close, vec![self_reg]);
    builder.ret(None);
}

/// Socket_read(self: *u8) -> *u8 (HaxeString*)
fn build_socket_read(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("sys_net_Socket_read")
        .param("self", ptr_u8.clone())
        .returns(ptr_u8)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let read = builder.get_function_by_name("rayzor_socket_read").unwrap();
    let result = builder.call(read, vec![self_reg]).unwrap();
    builder.ret(Some(result));
}

/// Socket_write(self: *u8, content: *u8)
fn build_socket_write(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("sys_net_Socket_write")
        .param("self", ptr_u8.clone())
        .param("content", ptr_u8)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let content = builder.get_param(1);
    let write = builder.get_function_by_name("rayzor_socket_write").unwrap();
    builder.call(write, vec![self_reg, content]);
    builder.ret(None);
}

/// Socket_shutdown(self: *u8, read: i32, write: i32)
fn build_socket_shutdown(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("sys_net_Socket_shutdown")
        .param("self", ptr_u8.clone())
        .param("read", i32_ty.clone())
        .param("write", i32_ty)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let read = builder.get_param(1);
    let write = builder.get_param(2);
    let shutdown = builder
        .get_function_by_name("rayzor_socket_shutdown")
        .unwrap();
    builder.call(shutdown, vec![self_reg, read, write]);
    builder.ret(None);
}

/// Socket_setBlocking(self: *u8, b: i32)
fn build_socket_set_blocking(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("sys_net_Socket_setBlocking")
        .param("self", ptr_u8.clone())
        .param("b", i32_ty)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let b = builder.get_param(1);
    let set = builder
        .get_function_by_name("rayzor_socket_set_blocking")
        .unwrap();
    builder.call(set, vec![self_reg, b]);
    builder.ret(None);
}

/// Socket_setTimeout(self: *u8, timeout: f64)
fn build_socket_set_timeout(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("sys_net_Socket_setTimeout")
        .param("self", ptr_u8.clone())
        .param("timeout", IrType::F64)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let timeout = builder.get_param(1);
    let set = builder
        .get_function_by_name("rayzor_socket_set_timeout")
        .unwrap();
    builder.call(set, vec![self_reg, timeout]);
    builder.ret(None);
}

/// Socket_setFastSend(self: *u8, b: i32)
fn build_socket_set_fast_send(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("sys_net_Socket_setFastSend")
        .param("self", ptr_u8.clone())
        .param("b", i32_ty)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let b = builder.get_param(1);
    let set = builder
        .get_function_by_name("rayzor_socket_set_fast_send")
        .unwrap();
    builder.call(set, vec![self_reg, b]);
    builder.ret(None);
}

/// Socket_waitForRead(self: *u8)
fn build_socket_wait_for_read(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("sys_net_Socket_waitForRead")
        .param("self", ptr_u8)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let wait = builder
        .get_function_by_name("rayzor_socket_wait_for_read")
        .unwrap();
    builder.call(wait, vec![self_reg]);
    builder.ret(None);
}

// =============================================================================
// Host MIR wrapper functions
// =============================================================================

/// Host_new(name: *u8) -> *u8
fn build_host_new(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("sys_net_Host_new")
        .param("name", ptr_u8.clone())
        .returns(ptr_u8)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let name = builder.get_param(0);
    let rt = builder.get_function_by_name("rayzor_host_new").unwrap();
    let result = builder.call(rt, vec![name]).unwrap();
    builder.ret(Some(result));
}

/// Host_toString(self: *u8) -> *u8
fn build_host_to_string(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("sys_net_Host_toString")
        .param("self", ptr_u8.clone())
        .returns(ptr_u8)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let rt = builder
        .get_function_by_name("rayzor_host_to_string")
        .unwrap();
    let result = builder.call(rt, vec![self_reg]).unwrap();
    builder.ret(Some(result));
}

/// Host_reverse(self: *u8) -> *u8
fn build_host_reverse(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("sys_net_Host_reverse")
        .param("self", ptr_u8.clone())
        .returns(ptr_u8)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_reg = builder.get_param(0);
    let rt = builder.get_function_by_name("rayzor_host_reverse").unwrap();
    let result = builder.call(rt, vec![self_reg]).unwrap();
    builder.ret(Some(result));
}

/// Host_localhost() -> *u8
fn build_host_localhost(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("sys_net_Host_localhost")
        .returns(ptr_u8)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let rt = builder
        .get_function_by_name("rayzor_host_localhost")
        .unwrap();
    let result = builder.call(rt, vec![]).unwrap();
    builder.ret(Some(result));
}
