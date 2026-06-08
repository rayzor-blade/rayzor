//! Quantisation scheme tags and per-block layout types.
//!
//! Q4KBlock / Q8Block are the internal decoded shapes used by the dequant +
//! INT8-encoded-X paths. Q4KMBlock / Q8KBlock are the public llama.cpp-canonical
//! layouts (`repr(C, packed(1))` for Q4_K_M, `repr(C)` for Q8_K) that match the
//! GGUF on-disk byte representation exactly.

/// Quantisation scheme tag. Mirrors the Haxe-side `QScheme` enum.
pub const QSCHEME_INT8: u8 = 0;
pub const QSCHEME_Q4_K_M: u8 = 1;
pub const QSCHEME_Q6_K: u8 = 2;

/// Q4_K_M block dimensions. Fixed by the GGUF spec.
pub const Q4_K_M_BLOCK_SIZE: usize = 256;
pub const Q4_K_M_BLOCK_BYTES: usize = 144;

/// Q6_K block dimensions. GGUF dtype 14 (token_embd, attn_v, ffn_down in K_M
/// variants). Layout (210 bytes): ql[128] + qh[64] + scales[16, i8] + d[f16].
pub const Q6_K_BLOCK_SIZE: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// In-memory view of a 144-byte Q4_K_M super-block. Decoded once into f32
/// scales + mins at dequant time so the inner kernel can stay arithmetic.
/// `d` and `dmin` are kept on the struct for diagnostics + unit tests even
/// though the hot kernel only consumes the already-scaled `scales`/`mins`.
pub struct Q4KBlock {
    #[allow(dead_code)]
    pub d: f32,
    #[allow(dead_code)]
    pub dmin: f32,
    /// Per-sub-block effective scale (already multiplied by `d`).
    pub scales: [f32; 8],
    /// Per-sub-block effective min (already multiplied by `dmin`).
    pub mins: [f32; 8],
    /// 256 nibbles packed into 128 bytes.
    pub quants: [u8; 128],
}

/// Per-super-block scratch produced by `quantize_x_block_q8`. One entry per
/// 256-element block of X. Internal layout used by the SDOT inner kernels.
pub struct Q8Block {
    /// 256 INT8 quants.
    pub quants: [i8; 256],
    /// F32 quantisation scale (`x[i] ≈ scale * quants[i]`).
    pub scale: f32,
    /// Σ of `quants[s*32 .. (s+1)*32]` per Q4_K_M-style 32-elem sub-block
    /// (8 entries). Used to fold the Q4_K_M min term.
    pub bsums: [i32; 8],
    /// Σ of `quants[s*16 .. (s+1)*16]` per Q6_K-style 16-elem sub-block
    /// (16 entries). Used to fold the Q6_K `-32` bias.
    pub bsums_16: [i32; 16],
}

/// llama.cpp-compatible `block_q8_K`. Layout MUST stay
/// `{ f32 + 256 × i8 + 16 × i16 }` (= 4 + 256 + 32 = 292 bytes) to allow
/// memcpy interop with GGUF-quantised Q8_K tensors.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Q8KBlock {
    /// Quantisation scale: `x[i] ≈ d * qs[i]`.
    pub d: f32,
    /// 256 INT8 quants.
    pub qs: [i8; 256],
    /// Per-16-element sum of `qs`. `bsums[s] = Σ_{j=0..16} qs[s*16 + j]`.
    /// Q4_K_M's vec-dot folds the dmin term as
    /// `dmin * x.d * (bsums[2*s] + bsums[2*s+1]) * mn6[s]`.
    pub bsums: [i16; 16],
}

/// llama.cpp-compatible `block_q4_K`. The struct itself is just a typed view
/// over the 144 on-disk bytes — the field layout matches the GGUF Q4_K_M
/// block byte-for-byte:
///
/// ```text
///   bytes  0..1     : f16 d        (super-block scale)
///   bytes  2..3     : f16 dmin     (super-block min)
///   bytes  4..15    : 12-byte packed (scale6, min6) header
///   bytes 16..143   : 128 nibbles × 2 = 256 quants
/// ```
///
/// `repr(C, packed(1))` so a `*const u8` cursor from a GGUF tensor can be
/// reinterpreted as `*const Q4KMBlock` without padding drift.
#[repr(C, packed(1))]
#[derive(Clone, Copy)]
pub struct Q4KMBlock {
    /// f16 super-block scale (raw bits).
    pub d: u16,
    /// f16 super-block min (raw bits).
    pub dmin: u16,
    /// 12-byte packed (6-bit scale, 6-bit min) × 8 sub-blocks.
    pub scales: [u8; 12],
    /// 128 bytes = 256 nibbles of 4-bit quants.
    pub qs: [u8; 128],
}

// Compile-time guard: keep the structs at their canonical llama.cpp sizes.
const _: () = {
    assert!(core::mem::size_of::<Q8KBlock>() == 4 + 256 + 32);
    assert!(core::mem::size_of::<Q4KMBlock>() == Q4_K_M_BLOCK_BYTES);
};
