//! NativeStackTrace runtime — Rust backtrace capture for haxe.NativeStackTrace
//!
//! Provides source-mapped backtrace capture at throw time, accessible via the NativeStackTrace API.
//! In debug mode, the compiler registers JIT function addresses with source metadata after
//! compilation. At throw time, raw backtrace IPs are resolved against this registry to produce
//! human-readable Haxe stack traces with function name, file, line, and column.

use crate::exception::get_exception_stack_trace;
use crate::haxe_array::HaxeArray;
use crate::haxe_string::HaxeString;
use std::alloc::{alloc, Layout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, RwLock};

// ============================================================================
// Debug/Production Mode Toggle
// ============================================================================

/// Global flag controlling whether stack traces are captured.
/// Set to true in debug mode (`rayzor run`), false in production (`rayzor run --release`).
static STACK_TRACES_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable stack trace capture at runtime.
/// Called by the compiler at startup based on debug/release mode.
#[no_mangle]
pub extern "C" fn rayzor_set_stack_traces_enabled(enabled: i32) {
    STACK_TRACES_ENABLED.store(enabled != 0, Ordering::Relaxed);
}

/// Check if stack traces are currently enabled.
pub fn is_enabled() -> bool {
    STACK_TRACES_ENABLED.load(Ordering::Relaxed)
}

// ============================================================================
// Function Source Registry
// ============================================================================

/// Source metadata for a JIT-compiled function.
/// Registered by the compiler after JIT finalization for debug-mode stack traces.
struct FunctionSourceInfo {
    /// Start address of the compiled function code
    code_start: usize,
    /// Fully qualified name (e.g., "Main.thrower", "haxe.Exception.new")
    qualified_name: String,
    /// Source file path (e.g., "test_exception.hx")
    source_file: String,
    /// Line number in source (1-based)
    line: u32,
    /// Column number in source (1-based)
    column: u32,
    /// The trimmed source line at `line`, pre-fetched at registration time.
    /// Empty string if the file could not be read or line is out of range.
    source_snippet: String,
}

/// Cache of already-read source files: path → lines.
/// Populated lazily at function registration time so throw-time formatting is free.
static SOURCE_FILE_CACHE: LazyLock<RwLock<std::collections::HashMap<String, Vec<String>>>> =
    LazyLock::new(|| RwLock::new(std::collections::HashMap::new()));

/// Cache of full source file contents: path → bytes.
/// Used by the ariadne renderer, which needs the raw source to draw
/// `(file_name, byte_range)` labels.
static SOURCE_CONTENT_CACHE: LazyLock<RwLock<std::collections::HashMap<String, String>>> =
    LazyLock::new(|| RwLock::new(std::collections::HashMap::new()));

/// Look up the full raw content of a source file, reading and caching on miss.
/// Returns empty string on failure.
fn fetch_source_content(path: &str) -> String {
    if path == "<unknown>" {
        return String::new();
    }
    if let Ok(cache) = SOURCE_CONTENT_CACHE.read() {
        if let Some(content) = cache.get(path) {
            return content.clone();
        }
    }
    let content = std::fs::read_to_string(path).unwrap_or_default();
    if !content.is_empty() {
        if let Ok(mut cache) = SOURCE_CONTENT_CACHE.write() {
            cache.insert(path.to_string(), content.clone());
        }
    }
    content
}

/// Resolve a (1-based line, 1-based column) pair to a byte offset within
/// `content`. Returns None when the position is out of range.
fn line_col_to_byte_offset(content: &str, line: u32, column: u32) -> Option<usize> {
    if line == 0 {
        return None;
    }
    let target_line = (line as usize).saturating_sub(1);
    let mut current_line = 0usize;
    let mut line_start = 0usize;
    for (idx, ch) in content.char_indices() {
        if current_line == target_line {
            let col = (column.saturating_sub(1)) as usize;
            return Some(line_start + col);
        }
        if ch == '\n' {
            current_line += 1;
            line_start = idx + 1;
        }
    }
    if current_line == target_line {
        let col = (column.saturating_sub(1)) as usize;
        return Some(line_start + col);
    }
    None
}

/// Look up a source line from the cache, reading the file if necessary.
/// Returns the raw (un-trimmed) line content, or empty string on failure.
fn fetch_source_line(path: &str, line: u32) -> String {
    if path == "<unknown>" || line == 0 {
        return String::new();
    }

    // Fast path: file already cached
    if let Ok(cache) = SOURCE_FILE_CACHE.read() {
        if let Some(lines) = cache.get(path) {
            return lines
                .get((line as usize).saturating_sub(1))
                .cloned()
                .unwrap_or_default();
        }
    }

    // Slow path: read and cache the file
    if let Ok(content) = std::fs::read_to_string(path) {
        let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
        let result = lines
            .get((line as usize).saturating_sub(1))
            .cloned()
            .unwrap_or_default();
        if let Ok(mut cache) = SOURCE_FILE_CACHE.write() {
            cache.insert(path.to_string(), lines);
        }
        result
    } else {
        String::new()
    }
}

/// Global registry of JIT function source info, keyed by compiler-assigned function ID.
static FUNCTION_REGISTRY: RwLock<Vec<FunctionSourceInfo>> = RwLock::new(Vec::new());

/// Maps compiler-assigned function ID → registry index for shadow stack lookups.
static FUNC_ID_MAP: LazyLock<RwLock<std::collections::HashMap<u32, u32>>> =
    LazyLock::new(|| RwLock::new(std::collections::HashMap::new()));

/// Register a JIT-compiled function's source metadata.
/// Called by the compiler after JIT finalization in debug mode.
/// `func_id` is the compiler-assigned IrFunctionId used by push_call_frame.
#[no_mangle]
pub extern "C" fn rayzor_register_function_source(
    func_id: u32,
    code_start: usize,
    qualified_name_ptr: *const u8,
    qualified_name_len: usize,
    source_file_ptr: *const u8,
    source_file_len: usize,
    line: u32,
    column: u32,
) {
    let qualified_name = if qualified_name_ptr.is_null() || qualified_name_len == 0 {
        String::from("<unknown>")
    } else {
        unsafe {
            String::from_utf8_lossy(std::slice::from_raw_parts(
                qualified_name_ptr,
                qualified_name_len,
            ))
            .into_owned()
        }
    };

    let source_file = if source_file_ptr.is_null() || source_file_len == 0 {
        String::from("<unknown>")
    } else {
        unsafe {
            String::from_utf8_lossy(std::slice::from_raw_parts(source_file_ptr, source_file_len))
                .into_owned()
        }
    };

    let source_snippet = fetch_source_line(&source_file, line);

    let info = FunctionSourceInfo {
        code_start,
        qualified_name,
        source_file,
        line,
        column,
        source_snippet,
    };

    if let Ok(mut registry) = FUNCTION_REGISTRY.write() {
        let index = registry.len() as u32;
        registry.push(info);
        // Map compiler func_id → registry index
        if let Ok(mut map) = FUNC_ID_MAP.write() {
            map.insert(func_id, index);
        }
    }
}

/// Look up source info for a given instruction pointer.
/// Returns the function whose code_start is the largest value <= ip.
fn lookup_function(ip: usize) -> Option<(String, String, u32, u32)> {
    let registry = FUNCTION_REGISTRY.read().ok()?;
    if registry.is_empty() {
        return None;
    }

    // Binary search for the largest code_start <= ip
    let idx = match registry.binary_search_by_key(&ip, |entry| entry.code_start) {
        Ok(idx) => idx,        // Exact match
        Err(0) => return None, // ip is before all functions
        Err(idx) => idx - 1,   // Largest code_start that is < ip
    };

    let info = &registry[idx];
    Some((
        info.qualified_name.clone(),
        info.source_file.clone(),
        info.line,
        info.column,
    ))
}

// ============================================================================
// Backtrace Resolution
// ============================================================================

/// Resolve a raw backtrace into a source-mapped stack trace string.
/// Each JIT frame that matches a registered function is formatted as:
///   Called from ClassName.method (file.hx line N column M)
/// Non-JIT frames (runtime, system) are skipped.
pub fn resolve_backtrace_to_source(bt: &backtrace::Backtrace) -> String {
    let registry = FUNCTION_REGISTRY.read().ok();
    let has_registry = registry.as_ref().is_some_and(|r| !r.is_empty());
    drop(registry); // Release lock before iterating

    if !has_registry {
        // No functions registered — return raw backtrace as fallback
        return format!("{:?}", bt);
    }

    let mut result = String::new();
    let mut seen = std::collections::HashSet::new();

    for frame in bt.frames() {
        let ip = frame.ip() as usize;
        if ip == 0 {
            continue;
        }

        if let Some((qualified_name, source_file, line, column)) = lookup_function(ip) {
            // Deduplicate (same function can appear multiple times due to inlining)
            let key = (qualified_name.clone(), line);
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);

            // Skip internal/generated functions
            if qualified_name.starts_with("__") {
                continue;
            }

            if !result.is_empty() {
                result.push('\n');
            }

            // Format: "Called from Class.method (file.hx line N column M)"
            result.push_str("Called from ");
            result.push_str(&qualified_name);

            // Only show file/line for user code (not stdlib internals)
            if line > 0 {
                result.push_str(" (");
                // Extract just the filename from the path
                let filename = source_file.rsplit('/').next().unwrap_or(&source_file);
                result.push_str(filename);
                result.push(':');
                result.push_str(&line.to_string());
                if column > 0 {
                    result.push(':');
                    result.push_str(&column.to_string());
                }
                result.push(')');
            }
        }
    }

    result
}

// ============================================================================
// Shadow Call Stack (debug mode instrumentation)
// ============================================================================

/// A single frame on the shadow call stack.
/// Stores the compiler-assigned func_id rather than the registry index so that
/// `rayzor_push_call_frame` is a cheap lock-free thread-local push.
/// The func_id → registry_index resolution happens only in `capture_shadow_stack`,
/// which is called exclusively at exception throw time.
struct ShadowFrame {
    /// Compiler-assigned IrFunctionId (resolved to FUNCTION_REGISTRY at capture time)
    func_id: u32,
    /// Runtime-updated line override (0 = use registry default)
    line: u32,
    /// Runtime-updated column override (0 = use registry default)
    column: u32,
}

thread_local! {
    static SHADOW_STACK: std::cell::RefCell<Vec<ShadowFrame>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Push a call frame onto the shadow stack (called at function entry in debug mode).
/// The `func_id` is the compiler-assigned IrFunctionId.
///
/// **Hot path**: this is called on every function entry in debug mode and must be as
/// cheap as possible.  We store `func_id` directly and defer the FUNC_ID_MAP lookup
/// (which requires a global RwLock) to `capture_shadow_stack`, which only runs when
/// an exception is thrown.
#[no_mangle]
pub extern "C" fn rayzor_push_call_frame(func_id: u32) {
    SHADOW_STACK.with(|stack| {
        stack.borrow_mut().push(ShadowFrame {
            func_id,
            line: 0,
            column: 0,
        });
    });
}

/// Pop a call frame from the shadow stack (called at function return in debug mode).
#[no_mangle]
pub extern "C" fn rayzor_pop_call_frame() {
    SHADOW_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

/// Return the current shadow stack depth.
/// Used by the exception runtime to snapshot depth on `push_handler`
/// so it can be restored on `longjmp` (otherwise frames between the
/// throw site and the catch landing pad leak permanently, since the
/// compiler only emits `pop_call_frame` on `Return` terminators).
pub fn shadow_stack_depth() -> usize {
    SHADOW_STACK.with(|stack| stack.borrow().len())
}

/// Truncate the shadow stack to `depth` frames.
/// Called by the exception runtime on the catch landing path immediately
/// before `_longjmp` so leaked frames from bypassed `Return` blocks are
/// dropped. Safe to call with `depth` >= current len (no-op).
pub fn truncate_shadow_stack(depth: usize) {
    SHADOW_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if depth < stack.len() {
            stack.truncate(depth);
        }
    });
}

/// Update the top frame's source location to reflect the current statement.
/// Emitted by the compiler before throw statements and call expressions so the
/// snapshot captures the exact line rather than the function definition line.
#[no_mangle]
pub extern "C" fn rayzor_update_call_frame_location(line: u32, column: u32) {
    SHADOW_STACK.with(|stack| {
        if let Some(frame) = stack.borrow_mut().last_mut() {
            frame.line = line;
            frame.column = column;
        }
    });
}

/// Capture the current shadow call stack as a formatted string.
/// Returns entries in reverse order (most recent call first).
///
/// Each frame is rendered through ariadne (the same formatter the
/// compiler diagnostics use), so exception stack traces are visually
/// consistent with compile-time errors. Frames whose source content
/// can't be read fall back to a single-line `Called from X (file:l:c)`
/// header so the trace still says something useful.
pub fn capture_shadow_stack() -> String {
    SHADOW_STACK.with(|stack| {
        let stack = stack.borrow();
        if stack.is_empty() {
            return String::new();
        }

        let registry = FUNCTION_REGISTRY.read().ok();
        let registry = match registry.as_ref() {
            Some(r) => r,
            None => return String::new(),
        };

        let mut result = String::new();
        let mut seen = std::collections::HashSet::new();

        // Resolve func_id → registry_index once, here at capture time (not at push time).
        let func_id_map = FUNC_ID_MAP.read().ok();

        // Iterate in reverse (most recent call first)
        for frame in stack.iter().rev() {
            let func_index = func_id_map
                .as_ref()
                .and_then(|m| m.get(&frame.func_id).copied())
                .unwrap_or(u32::MAX);
            if let Some(info) = registry.get(func_index as usize) {
                // Skip internal/generated functions
                if info.qualified_name.starts_with("__") {
                    continue;
                }

                // Prefer runtime-updated line/col over the registration-time defaults
                let line = if frame.line > 0 {
                    frame.line
                } else {
                    info.line
                };
                let column = if frame.line > 0 {
                    frame.column
                } else {
                    info.column
                };

                // Deduplicate by (function, line) so the same statement isn't shown twice
                let key = (func_index, line);
                if seen.contains(&key) {
                    continue;
                }
                seen.insert(key);

                if !result.is_empty() {
                    result.push('\n');
                }

                let frame_str = render_frame_ariadne(info, line, column);
                result.push_str(&frame_str);
            }
        }

        result
    })
}

/// Render a single stack frame through ariadne. Falls back to a plain
/// `Called from X (file:line:col)` header when the source file can't be
/// read (e.g. for stdlib functions whose sources aren't on disk relative
/// to the cwd).
fn render_frame_ariadne(info: &FunctionSourceInfo, line: u32, column: u32) -> String {
    use ariadne::{Color, Config, Label, Report, ReportKind, Source};

    let filename: &str = info
        .source_file
        .rsplit('/')
        .next()
        .unwrap_or(&info.source_file);

    let header = if line > 0 && column > 0 {
        format!(
            "Called from {} ({}:{}:{})",
            info.qualified_name, filename, line, column
        )
    } else if line > 0 {
        format!(
            "Called from {} ({}:{})",
            info.qualified_name, filename, line
        )
    } else {
        format!("Called from {}", info.qualified_name)
    };

    // Without a resolvable line we can't draw a frame; return the header
    // and let the caller stack frames in plain text.
    if line == 0 {
        return header;
    }

    let content = fetch_source_content(&info.source_file);
    if content.is_empty() {
        // Source file unavailable — keep at least the header + snippet line
        // that was pre-cached at registration time so the trace still says
        // *something* about the call site.
        let snippet = if !info.source_snippet.is_empty() {
            info.source_snippet.clone()
        } else {
            fetch_source_line(&info.source_file, line)
        };
        if snippet.is_empty() {
            return header;
        }
        return format!("{}\n   {} | {}", header, line, snippet.trim_end());
    }

    let offset = match line_col_to_byte_offset(&content, line, column.max(1)) {
        Some(o) => o.min(content.len().saturating_sub(1)),
        None => return header,
    };
    let end = (offset + 1).min(content.len());

    let label = format!("Called from {}", info.qualified_name);

    let mut builder = Report::<(&str, std::ops::Range<usize>)>::build(
        ReportKind::Custom("trace", Color::Cyan),
        filename,
        offset,
    )
    .with_config(Config::default().with_color(true).with_compact(true));
    builder = builder.with_label(
        Label::new((filename, offset..end))
            .with_message(label)
            .with_color(Color::Cyan),
    );

    let report = builder.finish();
    let mut buf: Vec<u8> = Vec::new();
    if report
        .write((filename, Source::from(content.as_str())), &mut buf)
        .is_err()
    {
        return header;
    }
    String::from_utf8(buf).unwrap_or(header)
}

// ============================================================================
// NativeStackTrace API (extern "C" functions mapped from Haxe stdlib)
// ============================================================================

/// Save the current stack trace for the given exception.
/// Called automatically at throw time. The exception parameter is a DynamicValue*
/// (Any = type-safe Dynamic = boxed value with type tag). Unused in our implementation
/// — we store the trace in thread-local ExceptionState.
#[no_mangle]
pub extern "C" fn rayzor_native_stack_trace_save_stack(_exception: *mut u8) {
    // Stack is already captured in rayzor_throw_typed, nothing extra needed
}

/// Helper: create a heap-allocated HaxeString from a Rust String.
pub fn make_haxe_string(s: String) -> *mut HaxeString {
    let bytes = s.into_bytes();
    let len = bytes.len();
    unsafe {
        let layout = Layout::from_size_align_unchecked(len + 1, 1);
        let ptr = alloc(layout);
        if ptr.is_null() {
            return std::ptr::null_mut();
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len);
        *ptr.add(len) = 0; // null terminator
        Box::into_raw(Box::new(HaxeString {
            ptr,
            len,
            cap: len + 1,
        }))
    }
}

/// Capture and return the current call stack as a HaxeString pointer.
/// In debug mode, returns source-mapped trace. In production, returns empty string.
#[no_mangle]
pub extern "C" fn rayzor_native_stack_trace_call_stack() -> *mut u8 {
    if is_enabled() {
        let trace_str = capture_shadow_stack();
        make_haxe_string(trace_str) as *mut u8
    } else {
        make_haxe_string(String::new()) as *mut u8
    }
}

/// Return the stored exception stack trace as a HaxeString pointer.
#[no_mangle]
pub extern "C" fn rayzor_native_stack_trace_exception_stack() -> *mut u8 {
    let trace_str = get_exception_stack_trace();
    make_haxe_string(trace_str) as *mut u8
}

/// Convert a native stack trace to a Haxe Array<StackItem>.
/// V1: returns an empty HaxeArray (full StackItem conversion deferred).
#[no_mangle]
pub extern "C" fn rayzor_native_stack_trace_to_haxe(_native_trace: *mut u8, _skip: i32) -> *mut u8 {
    // Return empty array — allocate on heap
    unsafe {
        let arr = Box::into_raw(Box::new(std::mem::zeroed::<HaxeArray>()));
        crate::haxe_array::haxe_array_new(arr, 8);
        arr as *mut u8
    }
}
