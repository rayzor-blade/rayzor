//! Haxe Math runtime implementation
//!
//! Standard math functions matching Haxe Math API

use std::f64::consts;

// ============================================================================
// Constants
// ============================================================================

/// Mathematical constant PI
#[no_mangle]
pub extern "C" fn haxe_math_pi() -> f64 {
    consts::PI
}

/// Mathematical constant E
#[no_mangle]
pub extern "C" fn haxe_math_e() -> f64 {
    consts::E
}

// ============================================================================
// Basic Math Operations
// ============================================================================

/// Absolute value
#[no_mangle]
pub extern "C" fn haxe_math_abs(x: f64) -> f64 {
    x.abs()
}

/// Minimum of two values
#[no_mangle]
pub extern "C" fn haxe_math_min(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// Maximum of two values
#[no_mangle]
pub extern "C" fn haxe_math_max(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Floor (round down)
#[no_mangle]
pub extern "C" fn haxe_math_floor(x: f64) -> i32 {
    x.floor() as i32
}

/// Ceiling (round up)
#[no_mangle]
pub extern "C" fn haxe_math_ceil(x: f64) -> i32 {
    x.ceil() as i32
}

/// Round to nearest integer
#[no_mangle]
pub extern "C" fn haxe_math_round(x: f64) -> i32 {
    x.round() as i32
}

/// Round to nearest integer, keeping the Float type (Haxe's Math.fround)
#[no_mangle]
pub extern "C" fn haxe_math_fround(x: f64) -> f64 {
    x.round()
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

/// Sine
#[no_mangle]
pub extern "C" fn haxe_math_sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine
#[no_mangle]
pub extern "C" fn haxe_math_cos(x: f64) -> f64 {
    x.cos()
}

/// Tangent
#[no_mangle]
pub extern "C" fn haxe_math_tan(x: f64) -> f64 {
    x.tan()
}

/// Arc sine
#[no_mangle]
pub extern "C" fn haxe_math_asin(x: f64) -> f64 {
    x.asin()
}

/// Arc cosine
#[no_mangle]
pub extern "C" fn haxe_math_acos(x: f64) -> f64 {
    x.acos()
}

/// Arc tangent
#[no_mangle]
pub extern "C" fn haxe_math_atan(x: f64) -> f64 {
    x.atan()
}

/// Arc tangent of y/x
#[no_mangle]
pub extern "C" fn haxe_math_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

// ============================================================================
// Exponential and Logarithmic Functions
// ============================================================================

/// Exponential (e^x)
#[no_mangle]
pub extern "C" fn haxe_math_exp(x: f64) -> f64 {
    x.exp()
}

/// Natural logarithm
#[no_mangle]
pub extern "C" fn haxe_math_log(x: f64) -> f64 {
    x.ln()
}

/// Power (x^y)
#[no_mangle]
pub extern "C" fn haxe_math_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// Square root
#[no_mangle]
pub extern "C" fn haxe_math_sqrt(x: f64) -> f64 {
    x.sqrt()
}

// ============================================================================
// Special Functions
// ============================================================================

/// Check if value is NaN
#[no_mangle]
pub extern "C" fn haxe_math_is_nan(x: f64) -> bool {
    x.is_nan()
}

/// Check if value is finite
#[no_mangle]
pub extern "C" fn haxe_math_is_finite(x: f64) -> bool {
    x.is_finite()
}

/// Random number between 0 and 1
#[no_mangle]
pub extern "C" fn haxe_math_random() -> f64 {
    // Simple LCG random number generator (not cryptographically secure)
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(1);

    let mut seed = SEED.load(Ordering::Relaxed);
    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    SEED.store(seed, Ordering::Relaxed);

    ((seed / 65536) % 32768) as f64 / 32768.0
}
