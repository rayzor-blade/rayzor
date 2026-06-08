//! Thin no_std-friendly shims over the `std::f32` math methods that move
//! to `libm` under `no_std`. Each function delegates to `libm` directly;
//! when the std prelude is in scope (test builds) the same shape works
//! without changes.

#[inline(always)]
pub fn roundf(x: f32) -> f32 {
    libm::roundf(x)
}
