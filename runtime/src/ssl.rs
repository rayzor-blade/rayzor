//! SSL/TLS runtime for sys.ssl.Socket, sys.ssl.Certificate, sys.ssl.Key, sys.ssl.Digest
//!
//! Backed by `rustls` (pure Rust TLS). Provides TLS client connections for HTTPS
//! and certificate management matching the Haxe stdlib interface.

use crate::haxe_string::HaxeString;
use crate::haxe_sys::HaxeBytes;
use rustls::pki_types::ServerName;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

/// Convert HaxeString pointer to owned Rust String
unsafe fn hs_to_rust(s_ptr: *const u8) -> Option<String> {
    if s_ptr.is_null() {
        return None;
    }
    let hs = &*(s_ptr as *const HaxeString);
    if hs.ptr.is_null() || hs.len == 0 {
        return Some(String::new());
    }
    let bytes = std::slice::from_raw_parts(hs.ptr, hs.len);
    Some(String::from_utf8_lossy(bytes).into_owned())
}

// ============================================================================
// SSL Socket
// ============================================================================

#[allow(dead_code)]
struct SslSocketHandle {
    tcp_stream: Option<TcpStream>,
    tls_conn: Option<rustls::ClientConnection>,
    tls_stream: Option<rustls::StreamOwned<rustls::ClientConnection, TcpStream>>,
    verify_cert: bool,
    hostname: Option<String>,
    ca_cert: Option<Arc<rustls::RootCertStore>>,
    blocking: bool,
    timeout: Option<std::time::Duration>,
}

/// Helper to get or create the TLS config
fn build_tls_config(handle: &SslSocketHandle) -> Result<rustls::ClientConfig, String> {
    let root_store = if let Some(ca) = &handle.ca_cert {
        (**ca).clone()
    } else {
        let mut store = rustls::RootCertStore::empty();
        store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        store
    };

    let config = if handle.verify_cert {
        rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth()
    } else {
        rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoVerifier))
            .with_no_client_auth()
    };

    Ok(config)
}

/// Certificate verifier that accepts anything (for verifyCert=false)
#[derive(Debug)]
struct NoVerifier;

impl rustls::client::danger::ServerCertVerifier for NoVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::aws_lc_rs::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// sys.ssl.Socket.new() -> handle
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_new() -> *mut u8 {
    let handle = Box::new(SslSocketHandle {
        tcp_stream: None,
        tls_conn: None,
        tls_stream: None,
        verify_cert: true,
        hostname: None,
        ca_cert: None,
        blocking: true,
        timeout: None,
    });
    Box::into_raw(handle) as *mut u8
}

/// sys.ssl.Socket.connect(host, port) — TCP connect + TLS handshake
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_connect(handle: *mut u8, host_ip: i32, port: i32) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);

        // Convert packed IPv4 to string
        let ip = format!(
            "{}.{}.{}.{}",
            (host_ip >> 24) & 0xFF,
            (host_ip >> 16) & 0xFF,
            (host_ip >> 8) & 0xFF,
            host_ip & 0xFF
        );

        let addr = format!("{}:{}", ip, port);
        let stream = match TcpStream::connect(&addr) {
            Ok(s) => s,
            Err(_) => return,
        };

        if let Some(timeout) = ssl.timeout {
            let _ = stream.set_read_timeout(Some(timeout));
            let _ = stream.set_write_timeout(Some(timeout));
        }

        let _ = stream.set_nonblocking(!ssl.blocking);

        // Determine SNI hostname — use configured hostname or IP
        let server_name = if let Some(ref hostname) = ssl.hostname {
            match ServerName::try_from(hostname.as_str()) {
                Ok(sn) => sn.to_owned(),
                Err(_) => {
                    // Fallback: try IP
                    match ServerName::try_from(ip.as_str()) {
                        Ok(sn) => sn.to_owned(),
                        Err(_) => return,
                    }
                }
            }
        } else {
            match ServerName::try_from(ip.as_str()) {
                Ok(sn) => sn.to_owned(),
                Err(_) => return,
            }
        };

        let config = match build_tls_config(ssl) {
            Ok(c) => c,
            Err(_) => return,
        };

        let tls_conn = match rustls::ClientConnection::new(Arc::new(config), server_name) {
            Ok(conn) => conn,
            Err(_) => return,
        };

        let tls_stream = rustls::StreamOwned::new(tls_conn, stream);
        ssl.tls_stream = Some(tls_stream);
    }
}

/// sys.ssl.Socket.handshake() — explicit handshake (TLS handshake happens on connect)
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_handshake(_handle: *mut u8) {
    // Handshake is performed during connect/first I/O in rustls
}

/// sys.ssl.Socket.setHostname(name)
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_set_hostname(handle: *mut u8, name: *const u8) {
    if handle.is_null() || name.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        if let Some(s) = hs_to_rust(name) {
            ssl.hostname = Some(s);
        }
    }
}

/// sys.ssl.Socket.setCA(cert)
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_set_ca(handle: *mut u8, cert_handle: *mut u8) {
    if handle.is_null() || cert_handle.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        let cert = &*(cert_handle as *const CertificateHandle);
        let mut store = rustls::RootCertStore::empty();
        for c in &cert.certs {
            let _ = store.add(c.clone());
        }
        ssl.ca_cert = Some(Arc::new(store));
    }
}

/// sys.ssl.Socket.setCertificate(cert, key) — client certificate
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_set_certificate(
    _handle: *mut u8,
    _cert: *mut u8,
    _key: *mut u8,
) {
    // Client certificate authentication — deferred for now
}

/// sys.ssl.Socket.peerCertificate() -> Certificate handle
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_peer_certificate(handle: *mut u8) -> *mut u8 {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let ssl = &*(handle as *const SslSocketHandle);
        if let Some(ref tls_stream) = ssl.tls_stream {
            if let Some(certs) = tls_stream.conn.peer_certificates() {
                if !certs.is_empty() {
                    let cert_handle = Box::new(CertificateHandle {
                        certs: certs.to_vec(),
                    });
                    return Box::into_raw(cert_handle) as *mut u8;
                }
            }
        }
        std::ptr::null_mut()
    }
}

/// sys.ssl.Socket.read() -> String
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_read(handle: *mut u8) -> *mut u8 {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        if let Some(ref mut tls_stream) = ssl.tls_stream {
            let mut buf = vec![0u8; 65536];
            match tls_stream.read(&mut buf) {
                Ok(n) if n > 0 => {
                    let s = String::from_utf8_lossy(&buf[..n]).to_string();
                    crate::ereg::rust_str_to_hs(&s)
                }
                _ => std::ptr::null_mut(),
            }
        } else {
            std::ptr::null_mut()
        }
    }
}

/// sys.ssl.Socket.write(data)
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_write(handle: *mut u8, data: *const u8) {
    if handle.is_null() || data.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        let data_str = hs_to_rust(data);
        if let Some(ref mut tls_stream) = ssl.tls_stream {
            if let Some(s) = data_str {
                let _ = tls_stream.write_all(s.as_bytes());
            }
        }
    }
}

/// sys.ssl.Socket.close()
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_close(handle: *mut u8) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        if let Some(ref mut tls_stream) = ssl.tls_stream {
            tls_stream.conn.send_close_notify();
        }
        ssl.tls_stream = None;
    }
}

/// sys.ssl.Socket.setBlocking(b)
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_set_blocking(handle: *mut u8, blocking: i32) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        ssl.blocking = blocking != 0;
        if let Some(ref tls_stream) = ssl.tls_stream {
            let _ = tls_stream.sock.set_nonblocking(!ssl.blocking);
        }
    }
}

/// sys.ssl.Socket.setTimeout(seconds)
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_set_timeout(handle: *mut u8, seconds: f64) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        if seconds <= 0.0 {
            ssl.timeout = None;
        } else {
            ssl.timeout = Some(std::time::Duration::from_secs_f64(seconds));
        }
        if let Some(ref tls_stream) = ssl.tls_stream {
            let _ = tls_stream.sock.set_read_timeout(ssl.timeout);
            let _ = tls_stream.sock.set_write_timeout(ssl.timeout);
        }
    }
}

/// sys.ssl.Socket.input -> handle (same SSL handle for I/O)
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_get_input(handle: *mut u8) -> *mut u8 {
    handle
}

/// sys.ssl.Socket.output -> handle (same SSL handle for I/O)
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_get_output(handle: *mut u8) -> *mut u8 {
    handle
}

/// SocketInput.readByte() for SSL socket
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_read_byte(handle: *mut u8) -> i32 {
    if handle.is_null() {
        return -1;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        if let Some(ref mut tls_stream) = ssl.tls_stream {
            let mut buf = [0u8; 1];
            match tls_stream.read_exact(&mut buf) {
                Ok(()) => buf[0] as i32,
                Err(_) => -1,
            }
        } else {
            -1
        }
    }
}

/// SocketInput.readBytes(bytes, pos, len) for SSL socket
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_read_bytes(
    handle: *mut u8,
    bytes: *mut u8,
    pos: i32,
    len: i32,
) -> i32 {
    if handle.is_null() || bytes.is_null() {
        return 0;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        let b = &mut *(bytes as *mut HaxeBytes);
        let pos = pos.max(0) as usize;
        let len = len.max(0) as usize;
        if pos + len > b.len {
            return 0;
        }
        if let Some(ref mut tls_stream) = ssl.tls_stream {
            let buf = std::slice::from_raw_parts_mut(b.ptr.add(pos), len);
            match tls_stream.read(buf) {
                Ok(n) => n as i32,
                Err(_) => 0,
            }
        } else {
            0
        }
    }
}

/// SocketOutput.writeByte(c) for SSL socket
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_write_byte(handle: *mut u8, c: i32) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        if let Some(ref mut tls_stream) = ssl.tls_stream {
            let _ = tls_stream.write_all(&[c as u8]);
        }
    }
}

/// SocketOutput.writeBytes(bytes, pos, len) for SSL socket
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_write_bytes(
    handle: *mut u8,
    bytes: *mut u8,
    pos: i32,
    len: i32,
) -> i32 {
    if handle.is_null() || bytes.is_null() {
        return 0;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        let b = &*(bytes as *const HaxeBytes);
        let pos = pos.max(0) as usize;
        let len = len.max(0) as usize;
        if pos + len > b.len {
            return 0;
        }
        if let Some(ref mut tls_stream) = ssl.tls_stream {
            let buf = std::slice::from_raw_parts(b.ptr.add(pos), len);
            match tls_stream.write(buf) {
                Ok(n) => n as i32,
                Err(_) => 0,
            }
        } else {
            0
        }
    }
}

/// SocketOutput.writeString(s) for SSL socket
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_write_string(handle: *mut u8, s: *const u8) {
    if handle.is_null() || s.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        let s_str = hs_to_rust(s);
        if let Some(ref mut tls_stream) = ssl.tls_stream {
            if let Some(s) = s_str {
                let _ = tls_stream.write_all(s.as_bytes());
            }
        }
    }
}

/// SocketOutput.flush() for SSL socket
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_flush(handle: *mut u8) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        if let Some(ref mut tls_stream) = ssl.tls_stream {
            let _ = tls_stream.flush();
        }
    }
}

/// sys.ssl.Socket.shutdown(read, write)
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_shutdown(handle: *mut u8, read: i32, write: i32) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let ssl = &mut *(handle as *mut SslSocketHandle);
        if let Some(ref tls_stream) = ssl.tls_stream {
            let how = match (read != 0, write != 0) {
                (true, true) => std::net::Shutdown::Both,
                (true, false) => std::net::Shutdown::Read,
                (false, true) => std::net::Shutdown::Write,
                _ => return,
            };
            let _ = tls_stream.sock.shutdown(how);
        }
    }
}

/// sys.ssl.Socket.setFastSend(b) — TCP_NODELAY
#[no_mangle]
pub extern "C" fn rayzor_ssl_socket_set_fast_send(handle: *mut u8, fast: i32) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let ssl = &*(handle as *const SslSocketHandle);
        if let Some(ref tls_stream) = ssl.tls_stream {
            let _ = tls_stream.sock.set_nodelay(fast != 0);
        }
    }
}

// ============================================================================
// Certificate
// ============================================================================

struct CertificateHandle {
    certs: Vec<rustls::pki_types::CertificateDer<'static>>,
}

/// Certificate.loadFile(path) -> Certificate
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_load_file(path: *const u8) -> *mut u8 {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let path_str = match hs_to_rust(path) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let file = match std::fs::File::open(&path_str) {
            Ok(f) => f,
            Err(_) => return std::ptr::null_mut(),
        };
        let mut reader = std::io::BufReader::new(file);
        let certs: Vec<_> = rustls_pemfile::certs(&mut reader)
            .filter_map(|r| r.ok())
            .collect();
        if certs.is_empty() {
            return std::ptr::null_mut();
        }
        Box::into_raw(Box::new(CertificateHandle { certs })) as *mut u8
    }
}

/// Certificate.loadPath(path) -> Certificate (loads all .pem/.crt files in directory)
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_load_path(path: *const u8) -> *mut u8 {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let path_str = match hs_to_rust(path) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let mut all_certs = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&path_str) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension()
                    .map(|e| e == "pem" || e == "crt")
                    .unwrap_or(false)
                {
                    if let Ok(file) = std::fs::File::open(&p) {
                        let mut reader = std::io::BufReader::new(file);
                        let certs: Vec<_> = rustls_pemfile::certs(&mut reader)
                            .filter_map(|r| r.ok())
                            .collect();
                        all_certs.extend(certs);
                    }
                }
            }
        }
        if all_certs.is_empty() {
            return std::ptr::null_mut();
        }
        Box::into_raw(Box::new(CertificateHandle { certs: all_certs })) as *mut u8
    }
}

/// Certificate.fromString(pem) -> Certificate
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_from_string(pem: *const u8) -> *mut u8 {
    if pem.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let pem_str = match hs_to_rust(pem) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let mut cursor = std::io::Cursor::new(pem_str.as_bytes());
        let certs: Vec<_> = rustls_pemfile::certs(&mut cursor)
            .filter_map(|r| r.ok())
            .collect();
        if certs.is_empty() {
            return std::ptr::null_mut();
        }
        Box::into_raw(Box::new(CertificateHandle { certs })) as *mut u8
    }
}

/// Certificate.loadDefaults() -> Certificate (system root CAs)
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_load_defaults() -> *mut u8 {
    // webpki-roots provides Mozilla's root CAs
    let certs: Vec<_> = webpki_roots::TLS_SERVER_ROOTS
        .iter()
        .map(|ta| rustls::pki_types::CertificateDer::from(ta.subject_public_key_info.to_vec()))
        .collect();
    Box::into_raw(Box::new(CertificateHandle { certs })) as *mut u8
}

/// Certificate.commonName -> String
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_common_name(_cert: *mut u8) -> *mut u8 {
    // X.509 parsing requires x509-parser crate — return empty for now
    crate::ereg::rust_str_to_hs("")
}

/// Certificate.altNames -> Array<String>
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_alt_names(_cert: *mut u8) -> *mut u8 {
    // Returns null — X.509 SAN parsing deferred
    std::ptr::null_mut()
}

/// Certificate.notBefore -> Date (as epoch float)
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_not_before(_cert: *mut u8) -> f64 {
    0.0
}

/// Certificate.notAfter -> Date (as epoch float)
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_not_after(_cert: *mut u8) -> f64 {
    0.0
}

/// Certificate.subject(field) -> String
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_subject(_cert: *mut u8, _field: *const u8) -> *mut u8 {
    std::ptr::null_mut()
}

/// Certificate.issuer(field) -> String
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_issuer(_cert: *mut u8, _field: *const u8) -> *mut u8 {
    std::ptr::null_mut()
}

/// Certificate.next() -> Certificate (chain traversal)
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_next(cert: *mut u8) -> *mut u8 {
    if cert.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let cert_handle = &*(cert as *const CertificateHandle);
        if cert_handle.certs.len() > 1 {
            let remaining = cert_handle.certs[1..].to_vec();
            Box::into_raw(Box::new(CertificateHandle { certs: remaining })) as *mut u8
        } else {
            std::ptr::null_mut()
        }
    }
}

/// Certificate.add(pem)
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_add(cert: *mut u8, pem: *const u8) {
    if cert.is_null() || pem.is_null() {
        return;
    }
    unsafe {
        let cert_handle = &mut *(cert as *mut CertificateHandle);
        let pem_str = match hs_to_rust(pem) {
            Some(s) => s,
            None => return,
        };
        let mut cursor = std::io::Cursor::new(pem_str.as_bytes());
        let certs: Vec<_> = rustls_pemfile::certs(&mut cursor)
            .filter_map(|r| r.ok())
            .collect();
        cert_handle.certs.extend(certs);
    }
}

/// Certificate.addDER(bytes)
#[no_mangle]
pub extern "C" fn rayzor_ssl_cert_add_der(cert: *mut u8, der_bytes: *mut u8) {
    if cert.is_null() || der_bytes.is_null() {
        return;
    }
    unsafe {
        let cert_handle = &mut *(cert as *mut CertificateHandle);
        let b = &*(der_bytes as *const HaxeBytes);
        let der_data = std::slice::from_raw_parts(b.ptr, b.len).to_vec();
        cert_handle
            .certs
            .push(rustls::pki_types::CertificateDer::from(der_data));
    }
}

// ============================================================================
// Key
// ============================================================================

struct KeyHandle {
    _key_der: Vec<u8>,
    _is_public: bool,
}

/// Key.loadFile(path, ?isPublic, ?pass) -> Key
#[no_mangle]
pub extern "C" fn rayzor_ssl_key_load_file(
    path: *const u8,
    is_public: i32,
    _pass: *const u8,
) -> *mut u8 {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let path_str = match hs_to_rust(path) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let data = match std::fs::read(&path_str) {
            Ok(d) => d,
            Err(_) => return std::ptr::null_mut(),
        };
        Box::into_raw(Box::new(KeyHandle {
            _key_der: data,
            _is_public: is_public != 0,
        })) as *mut u8
    }
}

/// Key.readPEM(data, isPublic, ?pass) -> Key
#[no_mangle]
pub extern "C" fn rayzor_ssl_key_read_pem(
    data: *const u8,
    is_public: i32,
    _pass: *const u8,
) -> *mut u8 {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let data_str = match hs_to_rust(data) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };
        let mut cursor = std::io::Cursor::new(data_str.as_bytes());
        let key_data = if is_public != 0 {
            // Try reading public key
            data_str.as_bytes().to_vec()
        } else {
            // Try reading private key
            match rustls_pemfile::private_key(&mut cursor) {
                Ok(Some(key)) => key.secret_der().to_vec(),
                _ => data_str.as_bytes().to_vec(),
            }
        };
        Box::into_raw(Box::new(KeyHandle {
            _key_der: key_data,
            _is_public: is_public != 0,
        })) as *mut u8
    }
}

/// Key.readDER(data, isPublic) -> Key
#[no_mangle]
pub extern "C" fn rayzor_ssl_key_read_der(data: *mut u8, is_public: i32) -> *mut u8 {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let b = &*(data as *const HaxeBytes);
        let der_data = std::slice::from_raw_parts(b.ptr, b.len).to_vec();
        Box::into_raw(Box::new(KeyHandle {
            _key_der: der_data,
            _is_public: is_public != 0,
        })) as *mut u8
    }
}

// ============================================================================
// Digest
// ============================================================================

/// Digest.make(data, algorithm) -> Bytes
/// Deferred: needs `ring` or `sha2` crate for hash computation.
/// HTTPS does not require this function — it's for explicit digest operations.
#[no_mangle]
pub extern "C" fn rayzor_ssl_digest_make(_data: *mut u8, _alg: *const u8) -> *mut u8 {
    std::ptr::null_mut()
}

/// Digest.sign(data, privKey, algorithm) -> Bytes
#[no_mangle]
pub extern "C" fn rayzor_ssl_digest_sign(
    _data: *mut u8,
    _priv_key: *mut u8,
    _alg: *const u8,
) -> *mut u8 {
    // Full signing requires key parsing — deferred
    std::ptr::null_mut()
}

/// Digest.verify(data, signature, pubKey, algorithm) -> Bool
#[no_mangle]
pub extern "C" fn rayzor_ssl_digest_verify(
    _data: *mut u8,
    _sig: *mut u8,
    _pub_key: *mut u8,
    _alg: *const u8,
) -> i32 {
    // Full verification requires key parsing — deferred
    0
}
