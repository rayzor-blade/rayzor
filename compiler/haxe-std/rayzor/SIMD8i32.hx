package rayzor;

/**
 * 256-bit SIMD vector of 8 × Int (i32) — the wide counterpart of `SIMD4i32`.
 *
 * The **accumulator** for 256-bit integer dot-product kernels: a widening dot
 * reduces 32 i8×i8 products into these 8 i32 lanes, which are then summed
 * horizontally. On x86-64 with AVX-VNNI the dot is one `vpdpbusd ymm`.
 *
 * ## Target support — this type is NOT portable
 * Available on the **LLVM** backend only; wasm (`v128`) and Cranelift have no
 * 256-bit vector type and refuse it rather than narrowing. Gate every use
 * behind `#if llvm` with a `SIMD4i32` fallback — see `SIMD32i8` for the shape.
 */
@:coreType
@:notNull
@:native("rayzor::SIMD8i32")
extern abstract SIMD8i32 {
    /** Broadcast a single value to all 8 lanes */
    @:native("splat")
    public static function splat(v:Int):SIMD8i32;

    /** Element-wise addition */
    @:native("add")
    @:op(A + B)
    public function add(other:SIMD8i32):SIMD8i32;

    /** Element-wise subtraction */
    @:native("sub")
    @:op(A - B)
    public function sub(other:SIMD8i32):SIMD8i32;

    /** Element-wise multiplication */
    @:native("mul")
    @:op(A * B)
    public function mul(other:SIMD8i32):SIMD8i32;

    /** Read lane: v[i] */
    @:arrayAccess
    @:native("extract")
    public function get(lane:Int):Int;

    /** Write lane: v[i] = x */
    @:arrayAccess
    @:native("insert")
    public function set(lane:Int, value:Int):SIMD8i32;

    /** Horizontal sum of all 8 lanes */
    @:native("sum")
    public function sum():Int;

    /**
     * Fused widening dot-accumulate: `acc + dot(a, b)` where the 32 i8×i8
     * products are summed in groups of 4 into the 8 i32 lanes. Signed × signed;
     * use `dotI8I7` for quantized-weight kernels whose RHS stays in i7 range.
     */
    @:native("dot")
    public static function dot(acc:SIMD8i32, a:SIMD32i8, b:SIMD32i8):SIMD8i32;

    /**
     * Quantized-weight dot-accumulate. Same result as `dot` when every lane of
     * `b` is in i7 range (0..63 for Q4/Q6 unpacked weights), but lets x86 use
     * the unsigned×signed VNNI instruction. Do not use for Q8×Q8 attention.
     */
    @:native("dotI8I7")
    public static function dotI8I7(acc:SIMD8i32, a:SIMD32i8, b:SIMD32i8):SIMD8i32;

    /**
     * Signed-lhs × unsigned-rhs dot-accumulate. `b` lanes are interpreted as
     * u8 (0..255). Used with shifted query bytes plus a block-sum correction.
     */
    @:native("dotI8U8")
    public static function dotI8U8(acc:SIMD8i32, a:SIMD32i8, b:SIMD32i8):SIMD8i32;
}
