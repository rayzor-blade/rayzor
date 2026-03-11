/// SSL/TLS type implementation using MIR Builder
///
/// Backs `sys.ssl.Socket`, `sys.ssl.Certificate`, `sys.ssl.Key`, `sys.ssl.Digest`
/// with Rust's `rustls` crate. All methods route to extern C runtime functions.
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// Build all SSL type functions
pub fn build_ssl_types(builder: &mut MirBuilder) {
    declare_ssl_socket_externs(builder);
    declare_ssl_cert_externs(builder);
    declare_ssl_key_externs(builder);
    declare_ssl_digest_externs(builder);
}

fn declare_ssl_socket_externs(builder: &mut MirBuilder) {
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
    let string_ty = IrType::String;
    let i32_ty = IrType::I32;
    let f64_ty = IrType::F64;
    let void_ty = IrType::Void;

    // rayzor_ssl_socket_new() -> handle
    let f = builder
        .begin_function("rayzor_ssl_socket_new")
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_connect(handle, host_ip, port)
    let f = builder
        .begin_function("rayzor_ssl_socket_connect")
        .param("handle", ptr_u8.clone())
        .param("host_ip", i32_ty.clone())
        .param("port", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_handshake(handle)
    let f = builder
        .begin_function("rayzor_ssl_socket_handshake")
        .param("handle", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_set_hostname(handle, name)
    let f = builder
        .begin_function("rayzor_ssl_socket_set_hostname")
        .param("handle", ptr_u8.clone())
        .param("name", string_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_set_ca(handle, cert)
    let f = builder
        .begin_function("rayzor_ssl_socket_set_ca")
        .param("handle", ptr_u8.clone())
        .param("cert", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_set_certificate(handle, cert, key)
    let f = builder
        .begin_function("rayzor_ssl_socket_set_certificate")
        .param("handle", ptr_u8.clone())
        .param("cert", ptr_u8.clone())
        .param("key", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_peer_certificate(handle) -> cert handle
    let f = builder
        .begin_function("rayzor_ssl_socket_peer_certificate")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_read(handle) -> string
    let f = builder
        .begin_function("rayzor_ssl_socket_read")
        .param("handle", ptr_u8.clone())
        .returns(string_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_write(handle, data)
    let f = builder
        .begin_function("rayzor_ssl_socket_write")
        .param("handle", ptr_u8.clone())
        .param("data", string_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_close(handle)
    let f = builder
        .begin_function("rayzor_ssl_socket_close")
        .param("handle", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_set_blocking(handle, blocking)
    let f = builder
        .begin_function("rayzor_ssl_socket_set_blocking")
        .param("handle", ptr_u8.clone())
        .param("blocking", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_set_timeout(handle, seconds)
    let f = builder
        .begin_function("rayzor_ssl_socket_set_timeout")
        .param("handle", ptr_u8.clone())
        .param("seconds", f64_ty)
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_get_input(handle) -> handle
    let f = builder
        .begin_function("rayzor_ssl_socket_get_input")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_get_output(handle) -> handle
    let f = builder
        .begin_function("rayzor_ssl_socket_get_output")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_shutdown(handle, read, write)
    let f = builder
        .begin_function("rayzor_ssl_socket_shutdown")
        .param("handle", ptr_u8.clone())
        .param("read", i32_ty.clone())
        .param("write", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_set_fast_send(handle, fast)
    let f = builder
        .begin_function("rayzor_ssl_socket_set_fast_send")
        .param("handle", ptr_u8.clone())
        .param("fast", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // I/O stream methods for SSL socket
    // rayzor_ssl_socket_read_byte(handle) -> i32
    let f = builder
        .begin_function("rayzor_ssl_socket_read_byte")
        .param("handle", ptr_u8.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_read_bytes(handle, bytes, pos, len) -> i32
    let f = builder
        .begin_function("rayzor_ssl_socket_read_bytes")
        .param("handle", ptr_u8.clone())
        .param("bytes", ptr_u8.clone())
        .param("pos", i32_ty.clone())
        .param("len", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_write_byte(handle, c)
    let f = builder
        .begin_function("rayzor_ssl_socket_write_byte")
        .param("handle", ptr_u8.clone())
        .param("c", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_write_bytes(handle, bytes, pos, len) -> i32
    let f = builder
        .begin_function("rayzor_ssl_socket_write_bytes")
        .param("handle", ptr_u8.clone())
        .param("bytes", ptr_u8.clone())
        .param("pos", i32_ty.clone())
        .param("len", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_write_string(handle, s)
    let f = builder
        .begin_function("rayzor_ssl_socket_write_string")
        .param("handle", ptr_u8.clone())
        .param("s", string_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_socket_flush(handle)
    let f = builder
        .begin_function("rayzor_ssl_socket_flush")
        .param("handle", ptr_u8.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);
}

fn declare_ssl_cert_externs(builder: &mut MirBuilder) {
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
    let string_ty = IrType::String;
    let f64_ty = IrType::F64;
    let void_ty = IrType::Void;

    for (name, params, ret) in [
        (
            "rayzor_ssl_cert_load_file",
            vec![("path", string_ty.clone())],
            ptr_u8.clone(),
        ),
        (
            "rayzor_ssl_cert_load_path",
            vec![("path", string_ty.clone())],
            ptr_u8.clone(),
        ),
        (
            "rayzor_ssl_cert_from_string",
            vec![("pem", string_ty.clone())],
            ptr_u8.clone(),
        ),
        ("rayzor_ssl_cert_load_defaults", vec![], ptr_u8.clone()),
        (
            "rayzor_ssl_cert_common_name",
            vec![("cert", ptr_u8.clone())],
            string_ty.clone(),
        ),
        (
            "rayzor_ssl_cert_alt_names",
            vec![("cert", ptr_u8.clone())],
            ptr_u8.clone(),
        ),
        (
            "rayzor_ssl_cert_not_before",
            vec![("cert", ptr_u8.clone())],
            f64_ty.clone(),
        ),
        (
            "rayzor_ssl_cert_not_after",
            vec![("cert", ptr_u8.clone())],
            f64_ty,
        ),
        (
            "rayzor_ssl_cert_subject",
            vec![("cert", ptr_u8.clone()), ("field", string_ty.clone())],
            string_ty.clone(),
        ),
        (
            "rayzor_ssl_cert_issuer",
            vec![("cert", ptr_u8.clone()), ("field", string_ty.clone())],
            string_ty.clone(),
        ),
        (
            "rayzor_ssl_cert_next",
            vec![("cert", ptr_u8.clone())],
            ptr_u8.clone(),
        ),
        (
            "rayzor_ssl_cert_add",
            vec![("cert", ptr_u8.clone()), ("pem", string_ty.clone())],
            void_ty.clone(),
        ),
        (
            "rayzor_ssl_cert_add_der",
            vec![("cert", ptr_u8.clone()), ("der", ptr_u8.clone())],
            void_ty,
        ),
    ] {
        let mut fb = builder.begin_function(name);
        for (pname, pty) in &params {
            fb = fb.param(*pname, pty.clone());
        }
        let f = fb
            .returns(ret)
            .calling_convention(CallingConvention::C)
            .build();
        builder.mark_as_extern(f);
    }
}

fn declare_ssl_key_externs(builder: &mut MirBuilder) {
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
    let string_ty = IrType::String;
    let i32_ty = IrType::I32;

    // rayzor_ssl_key_load_file(path, is_public, pass) -> key
    let f = builder
        .begin_function("rayzor_ssl_key_load_file")
        .param("path", string_ty.clone())
        .param("is_public", i32_ty.clone())
        .param("pass", string_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_key_read_pem(data, is_public, pass) -> key
    let f = builder
        .begin_function("rayzor_ssl_key_read_pem")
        .param("data", string_ty.clone())
        .param("is_public", i32_ty.clone())
        .param("pass", string_ty)
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_key_read_der(data, is_public) -> key
    let f = builder
        .begin_function("rayzor_ssl_key_read_der")
        .param("data", ptr_u8.clone())
        .param("is_public", i32_ty)
        .returns(ptr_u8)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);
}

fn declare_ssl_digest_externs(builder: &mut MirBuilder) {
    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
    let string_ty = IrType::String;
    let i32_ty = IrType::I32;

    // rayzor_ssl_digest_make(data, alg) -> bytes
    let f = builder
        .begin_function("rayzor_ssl_digest_make")
        .param("data", ptr_u8.clone())
        .param("alg", string_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_digest_sign(data, key, alg) -> bytes
    let f = builder
        .begin_function("rayzor_ssl_digest_sign")
        .param("data", ptr_u8.clone())
        .param("key", ptr_u8.clone())
        .param("alg", string_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);

    // rayzor_ssl_digest_verify(data, sig, key, alg) -> bool
    let f = builder
        .begin_function("rayzor_ssl_digest_verify")
        .param("data", ptr_u8.clone())
        .param("sig", ptr_u8.clone())
        .param("key", ptr_u8)
        .param("alg", string_ty)
        .returns(i32_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(f);
}
