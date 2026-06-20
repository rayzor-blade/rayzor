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
}
