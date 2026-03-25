//! TinyCC runtime API for Haxe — wraps libtcc FFI for runtime C compilation.
//!
//! These functions are registered as runtime symbols and called from compiled
//! Haxe code via the `rayzor.runtime.CC` extern class.
//!
//! TCC is statically linked into the compiler crate via build.rs. Since the
//! compiler and runtime link into the same binary, these extern declarations
//! resolve at link time.

use std::ffi::{c_char, CString};
use std::ptr;

use crate::haxe_string::HaxeString;

// ============================================================================
// TCC FFI declarations (resolved from compiler's statically linked libtcc)
// ============================================================================

#[allow(non_camel_case_types)]
type TCCState = std::ffi::c_void;

extern "C" {
    fn tcc_new() -> *mut TCCState;
    fn tcc_delete(s: *mut TCCState);
    fn tcc_set_lib_path(s: *mut TCCState, path: *const c_char);
    fn tcc_set_output_type(s: *mut TCCState, output_type: i32) -> i32;
    fn tcc_set_options(s: *mut TCCState, str: *const c_char) -> i32;
    fn tcc_compile_string(s: *mut TCCState, buf: *const c_char) -> i32;
    fn tcc_add_sysinclude_path(s: *mut TCCState, pathname: *const c_char) -> i32;
    fn tcc_add_include_path(s: *mut TCCState, pathname: *const c_char) -> i32;
    fn tcc_add_symbol(s: *mut TCCState, name: *const c_char, val: *const std::ffi::c_void) -> i32;
    fn tcc_add_file(s: *mut TCCState, filename: *const c_char) -> i32;
    fn tcc_relocate(s: *mut TCCState) -> i32;
    fn tcc_get_symbol(s: *mut TCCState, name: *const c_char) -> *mut std::ffi::c_void;

    // dlopen for loading frameworks and shared libraries at runtime
    fn dlopen(filename: *const c_char, flags: i32) -> *mut std::ffi::c_void;
}

const RTLD_LAZY: i32 = 0x1;

const TCC_OUTPUT_MEMORY: i32 = 1;

// ============================================================================
// System path discovery
// ============================================================================

/// Discover the macOS SDK path. Cached via OnceLock.
#[cfg(target_os = "macos")]
fn discover_macos_sdk() -> &'static Option<String> {
    static SDK: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    SDK.get_or_init(|| {
        let candidates = [
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
        ];
        for sdk in &candidates {
            if std::path::Path::new(sdk).is_dir() {
                return Some(sdk.to_string());
            }
        }
        None
    })
}

/// Discover system include paths for the current platform.
/// Results are cached via OnceLock so filesystem probing runs at most once per process.
fn discover_system_include_paths() -> &'static Vec<String> {
    static PATHS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    PATHS.get_or_init(|| {
        let mut paths = Vec::new();

        #[cfg(target_os = "macos")]
        {
            if let Some(sdk) = discover_macos_sdk() {
                let inc = format!("{}/usr/include", sdk);
                if std::path::Path::new(&inc).is_dir() {
                    paths.push(inc);
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            let candidates = ["/usr/include", "/usr/local/include"];
            for p in &candidates {
                if std::path::Path::new(p).is_dir() {
                    paths.push(p.to_string());
                }
            }
        }

        paths
    })
}

// ============================================================================
// Helper: extract a null-terminated CString from a HaxeString pointer
// ============================================================================

unsafe fn haxe_string_to_cstring(s: *const HaxeString) -> Option<CString> {
    if s.is_null() {
        return None;
    }
    let hs = &*s;
    if hs.ptr.is_null() || hs.len == 0 {
        return Some(CString::new("").unwrap());
    }
    let slice = std::slice::from_raw_parts(hs.ptr, hs.len);
    CString::new(slice).ok()
}

// ============================================================================
// Runtime API functions (called from Haxe via CC extern class)
// ============================================================================

/// Create a new TCC compilation context with output type set to memory.
/// Returns an opaque pointer to TCCState.
#[no_mangle]
pub extern "C" fn rayzor_tcc_create() -> *mut TCCState {
    unsafe {
        let state = tcc_new();
        if state.is_null() {
            return ptr::null_mut();
        }

        // Set TCC lib path for runtime include resolution
        let tcc_dir = std::env::var("RAYZOR_TCC_DIR").unwrap_or_else(|_| {
            let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            manifest
                .join("../compiler/vendor/tinycc")
                .to_string_lossy()
                .into_owned()
        });
        if let Ok(path) = CString::new(tcc_dir.as_str()) {
            tcc_set_lib_path(state, path.as_ptr());
        }

        // -nostdlib: prevent TCC from trying to load system library files
        // (macOS .tbd stubs are incompatible with TCC's linker).
        // Symbol resolution still works via dlsym(RTLD_DEFAULT) during
        // tcc_relocate — patched in tccelf.c to allow this in memory mode.
        let opts = CString::new("-nostdlib").unwrap();
        tcc_set_options(state, opts.as_ptr());
        tcc_set_output_type(state, TCC_OUTPUT_MEMORY);

        // Add sysinclude path AFTER set_output_type so tccdefs.h can be found.
        // TCC injects `#include <tccdefs.h>` internally during preprocessing.
        let inc_path = std::path::Path::new(&tcc_dir).join("include");
        if let Ok(cinc) = CString::new(inc_path.to_string_lossy().as_ref()) {
            tcc_add_sysinclude_path(state, cinc.as_ptr());
        }

        // Auto-discover and add system include paths
        for path in discover_system_include_paths() {
            if let Ok(cpath) = CString::new(path.as_str()) {
                tcc_add_sysinclude_path(state, cpath.as_ptr());
            }
        }

        state
    }
}

/// Compile a C source string.
/// Takes the TCC state and a HaxeString pointer to the source code.
/// Returns 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn rayzor_tcc_compile(state: *mut TCCState, code: *const HaxeString) -> i32 {
    if state.is_null() {
        return 0;
    }
    unsafe {
        let c_code = match haxe_string_to_cstring(code) {
            Some(s) => s,
            None => return 0,
        };
        let ret = tcc_compile_string(state, c_code.as_ptr());
        if ret < 0 {
            panic!("TCC compilation failed");
        }
        1
    }
}

/// Register a symbol (name → value) in the TCC context.
/// The value is an i64 that C code can reference via `extern`.
/// Takes the TCC state, a HaxeString pointer to the name, and the raw value.
#[no_mangle]
pub extern "C" fn rayzor_tcc_add_symbol(state: *mut TCCState, name: *const HaxeString, value: i64) {
    if state.is_null() {
        return;
    }
    unsafe {
        let c_name = match haxe_string_to_cstring(name) {
            Some(s) => s,
            None => return,
        };
        tcc_add_symbol(state, c_name.as_ptr(), value as *const std::ffi::c_void);
    }
}

/// Add a symbol whose value is stored on the heap so TCC can read it via `extern long`.
/// `tcc_add_symbol` maps a name to an *address* — for `extern long __arg0`,
/// TCC reads the long at that address. So we Box the value and leak it.
/// Returns the heap address (caller should free with rayzor_tcc_free_value after execution).
#[no_mangle]
pub extern "C" fn rayzor_tcc_add_value_symbol(
    state: *mut TCCState,
    name: *const HaxeString,
    value: i64,
) -> i64 {
    if state.is_null() {
        return 0;
    }
    unsafe {
        let c_name = match haxe_string_to_cstring(name) {
            Some(s) => s,
            None => return 0,
        };
        let boxed = Box::new(value);
        let ptr = Box::into_raw(boxed);
        tcc_add_symbol(state, c_name.as_ptr(), ptr as *const std::ffi::c_void);
        ptr as i64
    }
}

/// Free a value allocated by rayzor_tcc_add_value_symbol.
#[no_mangle]
pub extern "C" fn rayzor_tcc_free_value(addr: i64) {
    if addr == 0 {
        return;
    }
    unsafe {
        let _ = Box::from_raw(addr as *mut i64);
    }
}

/// Relocate all compiled code into executable memory.
/// Must be called after all compile() and addSymbol() calls.
/// Returns 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn rayzor_tcc_relocate(state: *mut TCCState) -> i32 {
    if state.is_null() {
        return 0;
    }
    unsafe {
        let ret = tcc_relocate(state);
        if ret < 0 {
            panic!("TCC relocation failed (undefined symbols or memory error)");
        }
        1
    }
}

/// Get a symbol address by name after relocation.
/// Returns the address as i64 (0 if not found).
#[no_mangle]
pub extern "C" fn rayzor_tcc_get_symbol(state: *mut TCCState, name: *const HaxeString) -> i64 {
    if state.is_null() {
        return 0;
    }
    unsafe {
        let c_name = match haxe_string_to_cstring(name) {
            Some(s) => s,
            None => return 0,
        };
        let sym = tcc_get_symbol(state, c_name.as_ptr());
        if sym.is_null() {
            let name_str = c_name.to_str().unwrap_or("<unknown>");
            panic!("TCC symbol not found: '{}'", name_str);
        }
        sym as i64
    }
}

/// Call a JIT-compiled function that takes no arguments and returns an i64.
/// On macOS ARM64 with MAP_JIT, TCC places static variables in the same
/// mmap region as code. After relocate() switches to executable mode,
/// writing to statics causes SIGBUS. We toggle write protection around
/// each JIT call so the code can both execute and write to its statics.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn jit_write_protect(enabled: bool) {
    extern "C" {
        fn pthread_jit_write_protect_np(enabled: i32);
    }
    pthread_jit_write_protect_np(if enabled { 1 } else { 0 });
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
unsafe fn jit_write_protect(_enabled: bool) {}

/// `fn_addr` is the address returned by `rayzor_tcc_get_symbol`.
#[no_mangle]
pub extern "C" fn rayzor_tcc_call0(fn_addr: i64) -> i64 {
    if fn_addr == 0 {
        return 0;
    }
    unsafe {
        let f: extern "C" fn() -> i64 = std::mem::transmute(fn_addr as usize);
        f()
    }
}

/// Call a JIT-compiled function with 1 i64 argument, returning i64.
#[no_mangle]
pub extern "C" fn rayzor_tcc_call1(fn_addr: i64, arg0: i64) -> i64 {
    if fn_addr == 0 {
        return 0;
    }
    unsafe {
        let f: extern "C" fn(i64) -> i64 = std::mem::transmute(fn_addr as usize);
        f(arg0)
    }
}

/// Call a JIT-compiled function with 2 i64 arguments, returning i64.
#[no_mangle]
pub extern "C" fn rayzor_tcc_call2(fn_addr: i64, arg0: i64, arg1: i64) -> i64 {
    if fn_addr == 0 {
        return 0;
    }
    unsafe {
        let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(fn_addr as usize);
        f(arg0, arg1)
    }
}

/// Call a JIT-compiled function with 3 i64 arguments, returning i64.
#[no_mangle]
pub extern "C" fn rayzor_tcc_call3(fn_addr: i64, arg0: i64, arg1: i64, arg2: i64) -> i64 {
    if fn_addr == 0 {
        return 0;
    }
    unsafe {
        let f: extern "C" fn(i64, i64, i64) -> i64 = std::mem::transmute(fn_addr as usize);
        f(arg0, arg1, arg2)
    }
}

/// Load a macOS framework or shared library into the TCC context.
///
/// For macOS frameworks (e.g. "Accelerate", "CoreFoundation"):
///   - Loads the framework dylib via dlopen so symbols are available
///   - Adds the framework's Headers/ directory as an include path
///   - After this, `#include <Accelerate/Accelerate.h>` works in C code
///
/// For shared libraries (e.g. "z", "sqlite3"):
///   - Loads libNAME.dylib (macOS) or libNAME.so (Linux) via dlopen
///
/// Returns 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn rayzor_tcc_add_framework(state: *mut TCCState, name: *const HaxeString) -> i32 {
    if state.is_null() {
        return 0;
    }
    unsafe {
        let fw_name = match haxe_string_to_cstring(name) {
            Some(s) => s,
            None => return 0,
        };
        let fw_str = match fw_name.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return 0,
        };

        // Try as macOS framework first
        #[cfg(target_os = "macos")]
        {
            let fw_dylib = format!("/System/Library/Frameworks/{}.framework/{}", fw_str, fw_str);
            if let Ok(c_path) = CString::new(fw_dylib.as_str()) {
                let handle = dlopen(c_path.as_ptr(), RTLD_LAZY);
                if !handle.is_null() {
                    // Framework loaded — add its Headers/ dir as include path.
                    // Use #include <Accelerate.h> (not <Accelerate/Accelerate.h>)
                    // since TCC doesn't support Apple's -iframework convention.
                    if let Some(sdk) = discover_macos_sdk() {
                        let fw_headers = format!(
                            "{}/System/Library/Frameworks/{}.framework/Headers",
                            sdk, fw_str
                        );
                        if std::path::Path::new(&fw_headers).is_dir() {
                            if let Ok(c_inc) = CString::new(fw_headers.as_str()) {
                                tcc_add_include_path(state, c_inc.as_ptr());
                            }
                        }
                    }
                    return 1;
                }
            }
        }

        // Fallback: try as shared library (libNAME.dylib / libNAME.so)
        #[cfg(target_os = "macos")]
        let lib_path = format!("lib{}.dylib", fw_str);
        #[cfg(not(target_os = "macos"))]
        let lib_path = format!("lib{}.so", fw_str);

        if let Ok(c_path) = CString::new(lib_path.as_str()) {
            let handle = dlopen(c_path.as_ptr(), RTLD_LAZY);
            if !handle.is_null() {
                return 1;
            }
        }

        0
    }
}

/// Add a directory to the include search path.
/// Returns 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn rayzor_tcc_add_include_path(
    state: *mut TCCState,
    path: *const HaxeString,
) -> i32 {
    if state.is_null() {
        return 0;
    }
    unsafe {
        let c_path = match haxe_string_to_cstring(path) {
            Some(s) => s,
            None => return 0,
        };
        let ret = tcc_add_include_path(state, c_path.as_ptr());
        if ret < 0 {
            0
        } else {
            1
        }
    }
}

/// Add a file (.c, .o, .a, .dylib, .so, .dll) to the TCC context.
/// C source files are compiled; object/archive/shared libs are linked.
/// Returns 1 on success, panics on failure.
#[no_mangle]
pub extern "C" fn rayzor_tcc_add_file(state: *mut TCCState, path: *const HaxeString) -> i32 {
    if state.is_null() {
        panic!("TCC addFile: null state");
    }
    unsafe {
        let c_path = match haxe_string_to_cstring(path) {
            Some(s) => s,
            None => panic!("TCC addFile: invalid path string"),
        };
        let ret = tcc_add_file(state, c_path.as_ptr());
        if ret < 0 {
            let path_str = c_path.to_str().unwrap_or("<unknown>");
            panic!("TCC addFile failed: '{}'", path_str);
        }
        1
    }
}

/// Add a system library via pkg-config discovery.
/// Runs `pkg-config --cflags --libs <name>` to discover include paths and
/// shared library locations. Adds include paths to TCC and loads the library.
/// Panics if pkg-config is not found or the library is not installed.
#[no_mangle]
pub extern "C" fn rayzor_tcc_add_clib(state: *mut TCCState, name: *const HaxeString) -> i32 {
    if state.is_null() {
        panic!("TCC addClib: null state");
    }
    let lib_name = unsafe {
        match haxe_string_to_cstring(name) {
            Some(s) => s.to_str().unwrap_or("").to_string(),
            None => panic!("TCC addClib: invalid library name"),
        }
    };

    // Run pkg-config --cflags to get include paths
    let cflags_output = std::process::Command::new("pkg-config")
        .args(["--cflags", &lib_name])
        .output();

    match cflags_output {
        Ok(output) if output.status.success() => {
            let cflags = String::from_utf8_lossy(&output.stdout);
            for flag in cflags.split_whitespace() {
                if let Some(path) = flag.strip_prefix("-I") {
                    if let Ok(c_path) = CString::new(path) {
                        unsafe {
                            tcc_add_include_path(state, c_path.as_ptr());
                        }
                    }
                }
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("pkg-config --cflags {} failed: {}", lib_name, stderr.trim());
        }
        Err(e) => {
            panic!(
                "pkg-config not found ({}). Install it via:\n  \
                 macOS: brew install pkg-config\n  \
                 Linux: apt install pkg-config\n  \
                 Windows/MSYS2: pacman -S pkg-config",
                e
            );
        }
    }

    // Run pkg-config --libs to get library flags, then dlopen the library
    let libs_output = std::process::Command::new("pkg-config")
        .args(["--libs", &lib_name])
        .output();

    match libs_output {
        Ok(output) if output.status.success() => {
            let libs = String::from_utf8_lossy(&output.stdout);
            let mut lib_dirs: Vec<String> = Vec::new();
            let mut lib_names: Vec<String> = Vec::new();

            for flag in libs.split_whitespace() {
                if let Some(dir) = flag.strip_prefix("-L") {
                    lib_dirs.push(dir.to_string());
                } else if let Some(l) = flag.strip_prefix("-l") {
                    lib_names.push(l.to_string());
                }
            }

            // Try to dlopen each library
            for l in &lib_names {
                // Try with explicit paths first, then system default
                let mut loaded = false;
                for dir in &lib_dirs {
                    #[cfg(target_os = "macos")]
                    let full_path = format!("{}/lib{}.dylib", dir, l);
                    #[cfg(not(target_os = "macos"))]
                    let full_path = format!("{}/lib{}.so", dir, l);

                    if let Ok(c_path) = CString::new(full_path.as_str()) {
                        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_LAZY) };
                        if !handle.is_null() {
                            loaded = true;
                            break;
                        }
                    }
                }
                if !loaded {
                    // Try system default path
                    #[cfg(target_os = "macos")]
                    let default_name = format!("lib{}.dylib", l);
                    #[cfg(not(target_os = "macos"))]
                    let default_name = format!("lib{}.so", l);

                    if let Ok(c_path) = CString::new(default_name.as_str()) {
                        unsafe {
                            dlopen(c_path.as_ptr(), RTLD_LAZY);
                        }
                    }
                }
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("pkg-config --libs {} failed: {}", lib_name, stderr.trim());
        }
        Err(_) => {} // Already handled above
    }

    1
}

/// Free the TCC compilation context.
/// Note: relocated code memory is intentionally leaked (JIT pattern).
#[no_mangle]
pub extern "C" fn rayzor_tcc_delete(state: *mut TCCState) {
    if state.is_null() {
        return;
    }
    unsafe {
        tcc_delete(state);
    }
}
