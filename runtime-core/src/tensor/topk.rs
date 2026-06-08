//! Helpers used by the top-k sampler's repetition-penalty scan.

/// Linear "is `target` in `recent`?" check. The Haxe-side window is typically
/// a 64-element ring buffer; on aarch64 we walk it 8 elements at a time via
/// `vceqq_s64` + horizontal OR, then a scalar tail for the remainder.
///
/// # Safety
/// `recent` must be a valid `&[i64]` slice (caller responsibility — we only
/// dereference `as_ptr()` + index, no out-of-bounds access).
#[inline(always)]
pub unsafe fn recent_contains(recent: &[i64], target: i64) -> bool {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use core::arch::aarch64::*;
        let target_v = vdupq_n_s64(target);
        let mut i = 0;
        let n = recent.len();
        let ptr = recent.as_ptr();
        while i + 8 <= n {
            let r0 = vld1q_s64(ptr.add(i));
            let r1 = vld1q_s64(ptr.add(i + 2));
            let r2 = vld1q_s64(ptr.add(i + 4));
            let r3 = vld1q_s64(ptr.add(i + 6));
            let m0 = vceqq_s64(r0, target_v);
            let m1 = vceqq_s64(r1, target_v);
            let m2 = vceqq_s64(r2, target_v);
            let m3 = vceqq_s64(r3, target_v);
            let combined = vorrq_u64(vorrq_u64(m0, m1), vorrq_u64(m2, m3));
            if vmaxvq_u32(vreinterpretq_u32_u64(combined)) != 0 {
                return true;
            }
            i += 8;
        }
        while i < n {
            if *ptr.add(i) == target {
                return true;
            }
            i += 1;
        }
        return false;
    }
    #[allow(unreachable_code)]
    recent.contains(&target)
}
