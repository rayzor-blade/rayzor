package rayzor;

/**
 * 128-bit SIMD vector of 4 × Int (i32).
 *
 * The integer companion to `SIMD4f`. A zero-cost `@:coreType` abstract that
 * maps directly to a native 128-bit SIMD register (i32x4 on SSE2 / NEON,
 * v128 on wasm). Arithmetic operators compile to single vector instructions.
 *
 * This is primarily the **accumulator** type for integer dot-product kernels
 * (e.g. quantized matmul): a widening dot reduces 16×i8 products into the 4
 * i32 lanes of a `SIMD4i32`, which is then horizontally summed.
 *
 * Example:
 * ```haxe
 * var a = SIMD4i32.make(1, 2, 3, 4);
 * var b = SIMD4i32.splat(10);
 * var c = a + b;        // [11, 12, 13, 14]
 * trace(c.get(2));      // 13
 * trace(c.sum());       // 50
 * ```
 */
@:coreType
@:notNull
@:native("rayzor::SIMD4i32")
extern abstract SIMD4i32 {
    /** Broadcast a single value to all 4 lanes */
    @:native("splat")
    public static function splat(v:Int):SIMD4i32;

    /** Construct from 4 individual values */
    @:native("make")
    public static function make(x:Int, y:Int, z:Int, w:Int):SIMD4i32;

    /** Element-wise addition */
    @:native("add")
    @:op(A + B)
    public function add(other:SIMD4i32):SIMD4i32;

    /** Element-wise subtraction */
    @:native("sub")
    @:op(A - B)
    public function sub(other:SIMD4i32):SIMD4i32;

    /** Element-wise multiplication */
    @:native("mul")
    @:op(A * B)
    public function mul(other:SIMD4i32):SIMD4i32;

    /** Read lane: v[i] */
    @:arrayAccess
    @:native("extract")
    public function get(lane:Int):Int;

    /** Write lane: v[i] = x */
    @:arrayAccess
    @:native("insert")
    public function set(lane:Int, value:Int):SIMD4i32;

    /** Horizontal sum of all 4 lanes */
    @:native("sum")
    public function sum():Int;

    /**
     * Fused widening dot-accumulate (the quantized-matmul primitive):
     * `acc + dot(a, b)` where the 16 i8×i8 products of `a` and `b` are summed
     * in groups of 4 into the 4 i32 lanes. This is signed i8 × signed i8;
     * use `dotI8I7` only for quantized-weight kernels whose RHS is known to
     * stay in i7 range.
     */
    @:native("dot")
    public static function dot(acc:SIMD4i32, a:SIMD16i8, b:SIMD16i8):SIMD4i32;

    /**
     * Quantized-weight dot-accumulate. Same mathematical result as `dot`
     * when every lane of `b` is in i7 range (0..63 for Q4/Q6 unpacked
     * weights), but gives x86 backends permission to use unsigned×signed
     * VNNI instructions. Do not use for Q8×Q8 attention.
     */
    @:native("dotI8I7")
    public static function dotI8I7(acc:SIMD4i32, a:SIMD16i8, b:SIMD16i8):SIMD4i32;

    /**
     * Signed-lhs × unsigned-rhs dot-accumulate. `b` lanes are interpreted as
     * u8 (0..255). Used by Q8 flash attention with shifted query bytes and a
     * separate block-sum correction.
     */
    @:native("dotI8U8")
    public static function dotI8U8(acc:SIMD4i32, a:SIMD16i8, b:SIMD16i8):SIMD4i32;
}
