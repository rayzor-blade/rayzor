//! Rotary position embedding (RoPE) helpers shared by native and WASM runtimes.
//!
//! The kernels here operate on contiguous F32 buffers. Target runtimes own
//! tensor allocation, dtype conversion, and FFI shape validation.

/// Apply Llama/GGUF interleaved RoPE to a contiguous tensor payload.
///
/// `x` and `out` are laid out as `[seq_len, num_heads, head_dim]`, flattened
/// row-major. `cos` and `sin` are `[max_seq_len, head_dim / 2]`. Adjacent
/// dimensions are paired: `(x[2i], x[2i + 1])`.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // kernel ABI: all dims/strides passed flat
pub fn apply_interleaved_f32(
    out: &mut [f32],
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    position_offset: usize,
) {
    if head_dim == 0 || !head_dim.is_multiple_of(2) || num_heads == 0 {
        return;
    }
    let half = head_dim / 2;
    let elements_per_row = num_heads * head_dim;
    debug_assert!(x.len() >= seq_len * elements_per_row);
    debug_assert!(out.len() >= seq_len * elements_per_row);

    let max_seq_len = cos.len() / half;
    debug_assert_eq!(sin.len() / half, max_seq_len);

    for s in 0..seq_len {
        let pos = s + position_offset;
        let row_base = s * elements_per_row;
        if pos >= max_seq_len {
            out[row_base..row_base + elements_per_row]
                .copy_from_slice(&x[row_base..row_base + elements_per_row]);
            continue;
        }

        for h in 0..num_heads {
            let head_base = row_base + h * head_dim;
            let table_base = pos * half;
            for i in 0..half {
                let c = cos[table_base + i];
                let sn = sin[table_base + i];
                let lo = head_base + 2 * i;
                let hi = lo + 1;
                let xlo = x[lo];
                let xhi = x[hi];
                out[lo] = xlo * c - xhi * sn;
                out[hi] = xlo * sn + xhi * c;
            }
        }
    }
}

/// Fill a `[max_seq_len, head_dim / 2]` F32 RoPE table.
///
/// `trig_fn` is supplied by the wrapper crate so native can use `f64::sin/cos`
/// while WASM can use `libm::{sin, cos}` without pulling `std` into
/// `runtime-core`.
#[inline(always)]
pub fn fill_table_f32<F: Fn(f64) -> f64>(
    out: &mut [f32],
    head_dim: usize,
    max_seq_len: usize,
    base: f64,
    trig_fn: F,
) {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return;
    }
    let half = head_dim / 2;
    debug_assert!(out.len() >= max_seq_len * half);
    let head_dim_f = head_dim as f64;
    for p in 0..max_seq_len {
        for i in 0..half {
            let theta = 1.0_f64 / libm::pow(base, (2 * i) as f64 / head_dim_f);
            out[p * half + i] = trig_fn((p as f64) * theta) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_rotation_uses_adjacent_pairs() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let cos = [0.0f32, 1.0];
        let sin = [1.0f32, 0.0];
        let mut out = [0.0f32; 4];

        apply_interleaved_f32(&mut out, &x, &cos, &sin, 1, 1, 4, 0);

        assert_eq!(out, [-2.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn table_matches_expected_base_schedule() {
        let mut table = [0.0f32; 4];
        fill_table_f32(&mut table, 4, 2, 10000.0, libm::cos);

        assert!((table[0] - 1.0).abs() < 1e-6);
        assert!((table[1] - 1.0).abs() < 1e-6);
        assert!((table[2] - libm::cos(1.0) as f32).abs() < 1e-6);
        assert!((table[3] - libm::cos(0.01) as f32).abs() < 1e-6);
    }
}
