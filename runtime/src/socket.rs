//! Networking runtime for sys.net.Socket and sys.net.Host
//!
//! Provides TCP socket operations (connect, bind, listen, accept, read, write)
//! and DNS host resolution. All functions use extern "C" ABI for JIT integration.

use crate::haxe_string::HaxeString;
use crate::haxe_sys::HaxeBytes;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::ptr;
use std::time::Duration;

// =============================================================================
// SocketHandle — opaque handle wrapping TcpStream/TcpListener
// =============================================================================

struct SocketHandle {
    stream: Option<TcpStream>,
    listener: Option<TcpListener>,
    blocking: bool,
    timeout: Option<Duration>,
}

// =============================================================================
// HostHandle — resolved DNS host
// =============================================================================
#[allow(dead_code)]
struct HostHandle {
    name: String,
    ip: u32, // packed IPv4 (network byte order)
}

// =============================================================================
// Helper: HaxeString ↔ Rust String conversion
// =============================================================================

unsafe fn haxe_string_to_rust(s_ptr: *const u8) -> String {
    if s_ptr.is_null() {
        return String::new();
    }
    let hs = &*(s_ptr as *const HaxeString);
    if hs.ptr.is_null() || hs.len == 0 {
        return String::new();
    }
    let bytes = std::slice::from_raw_parts(hs.ptr, hs.len);
    String::from_utf8_lossy(bytes).into_owned()
}

fn rust_string_to_haxe(s: &str) -> *mut u8 {
    let hs_ptr = crate::haxe_sys::haxe_string_from_string(s.as_ptr(), s.len());
    hs_ptr as *mut u8
}

fn ipv4_to_u32(ip: Ipv4Addr) -> u32 {
    let octets = ip.octets();
    ((octets[0] as u32) << 24)
        | ((octets[1] as u32) << 16)
        | ((octets[2] as u32) << 8)
        | (octets[3] as u32)
}

fn u32_to_ipv4(ip: u32) -> Ipv4Addr {
    Ipv4Addr::new(
        ((ip >> 24) & 0xFF) as u8,
        ((ip >> 16) & 0xFF) as u8,
        ((ip >> 8) & 0xFF) as u8,
        (ip & 0xFF) as u8,
    )
}

// =============================================================================
// Socket API
// =============================================================================

/// Create a new unconnected socket handle.
#[no_mangle]
pub extern "C" fn rayzor_socket_new() -> *mut u8 {
    let handle = Box::new(SocketHandle {
        stream: None,
        listener: None,
        blocking: true,
        timeout: None,
    });
    Box::into_raw(handle) as *mut u8
}

/// Connect socket to host:port. host_ip is packed IPv4 from HostHandle.
#[no_mangle]
pub extern "C" fn rayzor_socket_connect(handle: *mut u8, host_ip: i32, port: i32) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };
    let ip = u32_to_ipv4(host_ip as u32);
    let addr = SocketAddr::new(ip.into(), port as u16);

    match TcpStream::connect(addr) {
        Ok(stream) => {
            stream.set_nonblocking(!sock.blocking).ok();
            if let Some(timeout) = sock.timeout {
                stream.set_read_timeout(Some(timeout)).ok();
                stream.set_write_timeout(Some(timeout)).ok();
            }
            sock.stream = Some(stream);
        }
        Err(e) => {
            eprintln!("Socket connect error: {}", e);
        }
    }
}

/// Bind socket to host:port for listening.
#[no_mangle]
pub extern "C" fn rayzor_socket_bind(handle: *mut u8, host_ip: i32, port: i32) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };
    let ip = u32_to_ipv4(host_ip as u32);
    let addr = SocketAddr::new(ip.into(), port as u16);

    match TcpListener::bind(addr) {
        Ok(listener) => {
            listener.set_nonblocking(!sock.blocking).ok();
            sock.listener = Some(listener);
        }
        Err(e) => {
            eprintln!("Socket bind error: {}", e);
        }
    }
}

/// Start listening for connections. Backlog is the max pending connections.
#[no_mangle]
pub extern "C" fn rayzor_socket_listen(_handle: *mut u8, _backlog: i32) {
    // TcpListener::bind() already starts listening in Rust.
    // The backlog is set at OS level via bind(). No-op here.
}

/// Accept an incoming connection. Returns new SocketHandle or null.
#[no_mangle]
pub extern "C" fn rayzor_socket_accept(handle: *mut u8) -> *mut u8 {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let sock = unsafe { &*(handle as *const SocketHandle) };

    if let Some(ref listener) = sock.listener {
        match listener.accept() {
            Ok((stream, _addr)) => {
                stream.set_nonblocking(!sock.blocking).ok();
                let new_handle = Box::new(SocketHandle {
                    stream: Some(stream),
                    listener: None,
                    blocking: sock.blocking,
                    timeout: sock.timeout,
                });
                Box::into_raw(new_handle) as *mut u8
            }
            Err(_) => ptr::null_mut(),
        }
    } else {
        ptr::null_mut()
    }
}

/// Close the socket.
#[no_mangle]
pub extern "C" fn rayzor_socket_close(handle: *mut u8) {
    if handle.is_null() {
        return;
    }
    // Drop the handle, which closes stream/listener
    let _ = unsafe { Box::from_raw(handle as *mut SocketHandle) };
}

/// Read all available data from socket. Returns HaxeString pointer.
#[no_mangle]
pub extern "C" fn rayzor_socket_read(handle: *mut u8) -> *mut u8 {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };

    if let Some(ref mut stream) = sock.stream {
        let mut buf = Vec::with_capacity(4096);
        let mut tmp = [0u8; 4096];

        loop {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&tmp[..n]);
                    // If we got less than buffer size, likely done
                    if n < tmp.len() {
                        break;
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }

        let s = String::from_utf8_lossy(&buf);
        rust_string_to_haxe(&s)
    } else {
        ptr::null_mut()
    }
}

/// Write string data to socket.
#[no_mangle]
pub extern "C" fn rayzor_socket_write(handle: *mut u8, data: *const u8) {
    if handle.is_null() || data.is_null() {
        return;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };
    let content = unsafe { haxe_string_to_rust(data) };

    if let Some(ref mut stream) = sock.stream {
        let _ = stream.write_all(content.as_bytes());
        let _ = stream.flush();
    }
}

/// Shutdown the socket for reading, writing, or both.
#[no_mangle]
pub extern "C" fn rayzor_socket_shutdown(handle: *mut u8, read: i32, write: i32) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &*(handle as *const SocketHandle) };

    if let Some(ref stream) = sock.stream {
        let how = match (read != 0, write != 0) {
            (true, true) => Shutdown::Both,
            (true, false) => Shutdown::Read,
            (false, true) => Shutdown::Write,
            (false, false) => return,
        };
        let _ = stream.shutdown(how);
    }
}

/// Set blocking mode. b=1 for blocking, b=0 for non-blocking.
#[no_mangle]
pub extern "C" fn rayzor_socket_set_blocking(handle: *mut u8, b: i32) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };
    sock.blocking = b != 0;

    if let Some(ref stream) = sock.stream {
        let _ = stream.set_nonblocking(!sock.blocking);
    }
    if let Some(ref listener) = sock.listener {
        let _ = listener.set_nonblocking(!sock.blocking);
    }
}

/// Set timeout in seconds (as f64). 0.0 means no timeout.
#[no_mangle]
pub extern "C" fn rayzor_socket_set_timeout(handle: *mut u8, seconds: f64) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };

    if seconds <= 0.0 {
        sock.timeout = None;
    } else {
        sock.timeout = Some(Duration::from_secs_f64(seconds));
    }

    if let Some(ref stream) = sock.stream {
        let _ = stream.set_read_timeout(sock.timeout);
        let _ = stream.set_write_timeout(sock.timeout);
    }
}

/// Set TCP_NODELAY (fast send). b=1 to enable, b=0 to disable.
#[no_mangle]
pub extern "C" fn rayzor_socket_set_fast_send(handle: *mut u8, b: i32) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &*(handle as *const SocketHandle) };

    if let Some(ref stream) = sock.stream {
        let _ = stream.set_nodelay(b != 0);
    }
}

/// Block until data is available for reading.
#[no_mangle]
pub extern "C" fn rayzor_socket_wait_for_read(handle: *mut u8) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &*(handle as *const SocketHandle) };

    if let Some(ref stream) = sock.stream {
        // Temporarily set blocking + read 0 bytes via peek
        let _ = stream.set_nonblocking(false);
        let mut buf = [0u8; 1];
        // peek blocks until data available
        let _ = std::net::TcpStream::peek(stream, &mut buf);
        // Restore non-blocking if it was set
        if !sock.blocking {
            let _ = stream.set_nonblocking(true);
        }
    }
}

/// Get peer address. Writes host_ip and port to out-params.
#[no_mangle]
pub extern "C" fn rayzor_socket_peer(handle: *mut u8, out_host: *mut i32, out_port: *mut i32) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &*(handle as *const SocketHandle) };

    if let Some(ref stream) = sock.stream {
        if let Ok(SocketAddr::V4(v4)) = stream.peer_addr() {
            unsafe {
                *out_host = ipv4_to_u32(*v4.ip()) as i32;
                *out_port = v4.port() as i32;
            }
        }
    }
}

/// Get local address. Writes host_ip and port to out-params.
#[no_mangle]
pub extern "C" fn rayzor_socket_host_info(handle: *mut u8, out_host: *mut i32, out_port: *mut i32) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &*(handle as *const SocketHandle) };

    let addr = if let Some(ref stream) = sock.stream {
        stream.local_addr().ok()
    } else if let Some(ref listener) = sock.listener {
        listener.local_addr().ok()
    } else {
        None
    };

    if let Some(SocketAddr::V4(v4)) = addr {
        unsafe {
            *out_host = ipv4_to_u32(*v4.ip()) as i32;
            *out_port = v4.port() as i32;
        }
    }
}

/// Socket.select() — wait for readability/writability on multiple sockets.
/// Uses poll() on macOS/Linux. Returns null for now (complex — can be implemented
/// as a follow-up with proper anonymous struct construction).
#[no_mangle]
pub extern "C" fn rayzor_socket_select(
    _read_arr: *const u8,
    _write_arr: *const u8,
    _others_arr: *const u8,
    _timeout: f64,
) -> *mut u8 {
    // TODO: Implement with libc::poll or similar
    // For now, return null — users should use Future.create() for async patterns
    ptr::null_mut()
}

// =============================================================================
// SocketInput / SocketOutput — byte-level I/O adapters
// =============================================================================
// These operate on the same SocketHandle pointer. Socket.input and Socket.output
// return the socket handle directly — no separate allocation needed.

/// Get the socket's input adapter (returns the socket handle itself).
#[no_mangle]
pub extern "C" fn rayzor_socket_get_input(handle: *mut u8) -> *mut u8 {
    handle
}

/// Get the socket's output adapter (returns the socket handle itself).
#[no_mangle]
pub extern "C" fn rayzor_socket_get_output(handle: *mut u8) -> *mut u8 {
    handle
}

/// Read a single byte from the socket. Returns -1 on EOF/error.
#[no_mangle]
pub extern "C" fn rayzor_socket_read_byte(handle: *mut u8) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };
    if let Some(ref mut stream) = sock.stream {
        let mut buf = [0u8; 1];
        match stream.read_exact(&mut buf) {
            Ok(()) => buf[0] as i32,
            Err(_) => -1,
        }
    } else {
        -1
    }
}

/// Read bytes into a Bytes buffer. Returns number of bytes actually read.
#[no_mangle]
pub extern "C" fn rayzor_socket_read_bytes(
    handle: *mut u8,
    bytes_ptr: *mut HaxeBytes,
    pos: i32,
    len: i32,
) -> i32 {
    if handle.is_null() || bytes_ptr.is_null() || pos < 0 || len <= 0 {
        return 0;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };
    if let Some(ref mut stream) = sock.stream {
        let b = unsafe { &mut *bytes_ptr };
        let pos = pos as usize;
        let len = len as usize;
        if pos + len > b.len {
            return 0;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(b.ptr.add(pos), len) };
        match stream.read(buf) {
            Ok(n) => n as i32,
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => 0,
            Err(_) => 0,
        }
    } else {
        0
    }
}

/// Write a single byte to the socket.
#[no_mangle]
pub extern "C" fn rayzor_socket_write_byte(handle: *mut u8, byte: i32) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };
    if let Some(ref mut stream) = sock.stream {
        let _ = stream.write_all(&[byte as u8]);
    }
}

/// Write bytes from a Bytes buffer to the socket. Returns number of bytes written.
#[no_mangle]
pub extern "C" fn rayzor_socket_write_bytes(
    handle: *mut u8,
    bytes_ptr: *mut HaxeBytes,
    pos: i32,
    len: i32,
) -> i32 {
    if handle.is_null() || bytes_ptr.is_null() || pos < 0 || len <= 0 {
        return 0;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };
    if let Some(ref mut stream) = sock.stream {
        let b = unsafe { &*bytes_ptr };
        let pos = pos as usize;
        let len = len as usize;
        if pos + len > b.len {
            return 0;
        }
        let buf = unsafe { std::slice::from_raw_parts(b.ptr.add(pos), len) };
        match stream.write(buf) {
            Ok(n) => n as i32,
            Err(_) => 0,
        }
    } else {
        0
    }
}

/// Write a string to the socket.
#[no_mangle]
pub extern "C" fn rayzor_socket_write_string(handle: *mut u8, str_ptr: *const u8) {
    if handle.is_null() || str_ptr.is_null() {
        return;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };
    let content = unsafe { haxe_string_to_rust(str_ptr) };
    if let Some(ref mut stream) = sock.stream {
        let _ = stream.write_all(content.as_bytes());
    }
}

/// Flush the socket output.
#[no_mangle]
pub extern "C" fn rayzor_socket_flush(handle: *mut u8) {
    if handle.is_null() {
        return;
    }
    let sock = unsafe { &mut *(handle as *mut SocketHandle) };
    if let Some(ref mut stream) = sock.stream {
        let _ = stream.flush();
    }
}

// =============================================================================
// Host API
// =============================================================================

/// Create a new Host by resolving a hostname or IP string.
#[no_mangle]
pub extern "C" fn rayzor_host_new(name_ptr: *const u8) -> *mut u8 {
    if name_ptr.is_null() {
        return ptr::null_mut();
    }
    let name = unsafe { haxe_string_to_rust(name_ptr) };

    // Try to resolve the hostname
    let ip = if let Ok(addr) = name.parse::<Ipv4Addr>() {
        // Direct IP address
        ipv4_to_u32(addr)
    } else {
        // DNS resolution
        let host_port = format!("{}:0", name);
        match host_port.to_socket_addrs() {
            Ok(mut addrs) => {
                if let Some(SocketAddr::V4(v4)) = addrs.find(|a| a.is_ipv4()) {
                    ipv4_to_u32(*v4.ip())
                } else {
                    0 // Resolution failed
                }
            }
            Err(_) => 0,
        }
    };

    let handle = Box::new(HostHandle { name, ip });
    Box::into_raw(handle) as *mut u8
}

/// Get the IP address as a packed i32.
#[no_mangle]
pub extern "C" fn rayzor_host_get_ip(handle: *const u8) -> i32 {
    if handle.is_null() {
        return 0;
    }
    let host = unsafe { &*(handle as *const HostHandle) };
    host.ip as i32
}

/// Get string representation of the host IP.
#[no_mangle]
pub extern "C" fn rayzor_host_to_string(handle: *const u8) -> *mut u8 {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let host = unsafe { &*(handle as *const HostHandle) };
    let ip = u32_to_ipv4(host.ip);
    let s = ip.to_string();
    rust_string_to_haxe(&s)
}

/// Reverse DNS lookup.
#[no_mangle]
pub extern "C" fn rayzor_host_reverse(handle: *const u8) -> *mut u8 {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let host = unsafe { &*(handle as *const HostHandle) };
    let ip = u32_to_ipv4(host.ip);

    // Attempt reverse DNS via lookup on the IP
    let addr = SocketAddr::new(ip.into(), 0);
    match dns_lookup_reverse(addr) {
        Some(name) => rust_string_to_haxe(&name),
        None => rust_string_to_haxe(&ip.to_string()),
    }
}

/// Get the local hostname.
#[no_mangle]
pub extern "C" fn rayzor_host_localhost() -> *mut u8 {
    let mut buf = [0u8; 256];
    let ret = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if ret == 0 {
        let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let s = String::from_utf8_lossy(&buf[..len]);
        rust_string_to_haxe(&s)
    } else {
        rust_string_to_haxe("localhost")
    }
}

// Reverse DNS helper — uses getaddrinfo style lookup
fn dns_lookup_reverse(addr: SocketAddr) -> Option<String> {
    // Simple reverse: format IP as string, try to resolve back
    // Full reverse DNS would need libc::getnameinfo, but for basic use:
    let ip_str = addr.ip().to_string();
    let host_port = format!("{}:0", ip_str);
    match host_port.to_socket_addrs() {
        Ok(_) => Some(ip_str), // Can't do true reverse without getnameinfo
        Err(_) => None,
    }
}
