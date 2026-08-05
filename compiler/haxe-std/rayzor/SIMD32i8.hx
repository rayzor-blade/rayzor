package rayzor;

/**
 * 256-bit SIMD vector of 32 × Int8 (i8) — the wide counterpart of `SIMD16i8`.
 *
 * The operands of `SIMD8i32.dotI8I7`, which processes 32 quantized weights per
 * instruction instead of 16. On x86-64 with AVX-VNNI that is a single
 * `vpdpbusd ymm`; on aarch64 it lowers to a pair of `sdot`s (correct, but no
 * faster than two `SIMD16i8` dots, since NEON has no 256-bit register).
 *
 * ## Target support — this type is NOT portable
 * 256-bit vectors exist only on the **LLVM** backend. wasm's SIMD is `v128`
 * and has no wider register at all (relaxed-SIMD did not widen it), and
 * Cranelift models only 128-bit vector types. Both REFUSE this type rather
 * than narrowing it, so a kernel that uses it must be gated:
 *
 * ```haxe
 * #if llvm
 *     var w = SIMD32i8.load(p);          // 32 bytes per step
 *     acc = SIMD8i32.dotI8I7(acc, a, w);
 * #else
 *     var w = SIMD16i8.load(p);          // 16 bytes per step
 *     acc4 = SIMD4i32.dotI8I7(acc4, a, w);
 * #end
 * ```
 *
 * Reach for it only in kernels where the wider step measurably pays; prefer
 * `SIMD16i8` everywhere else so the code stays portable.
 */
@:coreType
@:notNull
@:native("rayzor::SIMD32i8")
extern abstract SIMD32i8 {
    /** Broadcast a single byte (low 8 bits of `v`) to all 32 lanes */
    @:native("splat")
    public static function splat(v:Int):SIMD32i8;

    /** Load 32 contiguous bytes from a pointer */
    @:native("load")
    public static function load(ptr:Ptr<Int>):SIMD32i8;

    /**
     * Bitwise AND of two i8x32 vectors, lane-wise. The mask for Q4 nibble
     * unpack: `SIMD32i8.and(v, SIMD32i8.splat(0x0F))` keeps the low nibble.
     */
    @:native("and")
    public static function and(a:SIMD32i8, b:SIMD32i8):SIMD32i8;

    /** Bitwise OR of two i8x32 vectors, lane-wise. */
    @:native("or")
    public static function or(a:SIMD32i8, b:SIMD32i8):SIMD32i8;

    /** Bitwise XOR of two i8x32 vectors, lane-wise. */
    @:native("xor")
    public static function xor(a:SIMD32i8, b:SIMD32i8):SIMD32i8;

    /** Logical left shift of every lane by `n` (same amount for all 32 lanes). */
    @:native("shl")
    public static function shl(a:SIMD32i8, n:Int):SIMD32i8;

    /**
     * Arithmetic (sign-propagating) right shift of every lane by `n`.
     * For zero-filling unpack (e.g. Q4 high nibble) use `ushr`.
     */
    @:native("shr")
    public static function shr(a:SIMD32i8, n:Int):SIMD32i8;

    /**
     * Logical (zero-filling) right shift of every lane by `n`. The Q4 high
     * nibble: `SIMD32i8.ushr(v, 4)` yields the top 4 bits as 0..15.
     */
    @:native("ushr")
    public static function ushr(a:SIMD32i8, n:Int):SIMD32i8;

    /**
     * Read lane `i` as an Int (the i8 value, sign-extended). Bytes are signed;
     * mask `& 0xFF` for an unsigned 0..255 value.
     */
    @:arrayAccess
    @:native("extract")
    public function get(lane:Int):Int;
}
