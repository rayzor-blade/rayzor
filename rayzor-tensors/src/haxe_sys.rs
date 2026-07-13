//! Plugin-local mirror of the Haxe `Bytes` representation. `#[repr(C)]` with a
//! layout identical to rayzor-runtime's `haxe_sys::HaxeBytes`, so buffers cross
//! the boundary by pointer safely. Kept minimal — only what the kernels touch.
//! (If the runtime layout changes, change it here too — the shared repr is the
//! contract.)

/// Haxe Bytes — raw byte buffer with mmap + view backends.
#[repr(C)]
pub struct HaxeBytes {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
    pub kind: u8,
    _pad: [u8; 7],
    pub owner: *mut HaxeBytes,
    pub refcount: i64,
    pub fd: i32,
    _pad2: [u8; 4],
}

pub const HAXE_BYTES_KIND_MALLOC: u8 = 0;

impl HaxeBytes {
    #[inline]
    pub fn new_malloc(ptr: *mut u8, len: usize, cap: usize) -> Self {
        HaxeBytes {
            ptr,
            len,
            cap,
            kind: HAXE_BYTES_KIND_MALLOC,
            _pad: [0; 7],
            owner: std::ptr::null_mut(),
            refcount: 1,
            fd: -1,
            _pad2: [0; 4],
        }
    }
}
