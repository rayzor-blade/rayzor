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
}
