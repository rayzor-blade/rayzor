package rayzor;

/**
 * 128-bit SIMD vector of 16 × Int8 (i8).
 *
 * These are the operands of the fused integer dot-product `SIMD4i32.dot`, the
 * quantized-matmul primitive: 16 i8 activations × 16 i8 (or i4/i6-unpacked)
 * weights, reduced in groups of 4 into a `SIMD4i32` accumulator. A zero-cost
 * `@:coreType` abstract over a native 128-bit register (i8x16 / v128).
 *
 * Example:
 * ```haxe
 * var acc = SIMD4i32.splat(0);
 * var a = SIMD16i8.load(activationPtr);
 * var b = SIMD16i8.load(weightPtr);
 * acc = SIMD4i32.dot(acc, a, b);   // acc += widening_dot(a, b)
 * trace(acc.sum());
 * ```
 */
@:coreType
@:notNull
@:native("rayzor::SIMD16i8")
extern abstract SIMD16i8 {
    /** Broadcast a single byte (low 8 bits of `v`) to all 16 lanes */
    @:native("splat")
    public static function splat(v:Int):SIMD16i8;

    /** Load 16 contiguous bytes from a pointer */
    @:native("load")
    public static function load(ptr:Ptr<Int>):SIMD16i8;

    /**
     * Bitwise AND of two i8x16 vectors, lane-wise. The mask for Q4 nibble
     * unpack: `SIMD16i8.and(v, SIMD16i8.splat(0x0F))` keeps the low nibble.
     */
    @:native("and")
    public static function and(a:SIMD16i8, b:SIMD16i8):SIMD16i8;

    /** Bitwise OR of two i8x16 vectors, lane-wise. */
    @:native("or")
    public static function or(a:SIMD16i8, b:SIMD16i8):SIMD16i8;

    /** Bitwise XOR of two i8x16 vectors, lane-wise. */
    @:native("xor")
    public static function xor(a:SIMD16i8, b:SIMD16i8):SIMD16i8;

    /** Logical left shift of every lane by `n` (same amount for all 16 lanes). */
    @:native("shl")
    public static function shl(a:SIMD16i8, n:Int):SIMD16i8;

    /**
     * Arithmetic (sign-propagating) right shift of every lane by `n`.
     * For zero-filling unpack (e.g. Q4 high nibble) use `ushr`.
     */
    @:native("shr")
    public static function shr(a:SIMD16i8, n:Int):SIMD16i8;

    /**
     * Logical (zero-filling) right shift of every lane by `n`. The Q4 high
     * nibble: `SIMD16i8.ushr(v, 4)` yields the top 4 bits as 0..15.
     */
    @:native("ushr")
    public static function ushr(a:SIMD16i8, n:Int):SIMD16i8;
}
