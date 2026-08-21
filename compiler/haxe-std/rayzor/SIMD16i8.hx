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

    /** Write the SIMD16i8 back to `ptr`. The integer vector types could be
        loaded but not stored, which left every quantise with a scalar write
        loop after a vectorised body. */
    @:native("store")
    public function store(ptr:Ptr<Int>):Void;

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

    /**
     * Byte-lane shuffle: lane `i` of the result is lane `idx[i]` of `a`.
     *
     * `idx[i]` must be 0..15 to select a lane, or have bit 7 set (e.g. 0x80) to
     * yield 0. Values 16..127 are TARGET-DEPENDENT and must not be used: x86
     * `pshufb` wraps to the low 4 bits while NEON and wasm zero. The contract is
     * the intersection of `pshufb` / `tbl1` / `i8x16.swizzle` so every backend
     * lowers to a single instruction; widening it to full wasm semantics would
     * cost x86 an extra `paddusb` in the hot loop.
     *
     * A mask built from `make16` literals folds to a constant, which x86 then
     * folds into `vpshufb`'s memory operand — costing no live register.
     */
    @:native("shuffle")
    public static function shuffle(a:SIMD16i8, idx:SIMD16i8):SIMD16i8;

    /**
     * Build a byte vector from 16 lane values (low 8 bits of each). Intended for
     * shuffle masks: an all-literal call constant-folds into rodata.
     */
    @:native("make16")
    public static function make16(b0:Int, b1:Int, b2:Int, b3:Int, b4:Int, b5:Int,
        b6:Int, b7:Int, b8:Int, b9:Int, b10:Int, b11:Int, b12:Int, b13:Int,
        b14:Int, b15:Int):SIMD16i8;

    /**
     * Read lane `i` as an Int (the i8 value, sign-extended). Lets a kernel pull
     * individual bytes out of a loaded vector in-guest — e.g. the Q4 block
     * header `d`/`dmin`/`scales` from the first 16 bytes. Bytes are signed; mask
     * `& 0xFF` for an unsigned 0..255 value.
     */
    @:arrayAccess
    @:native("extract")
    public function get(lane:Int):Int;
}
