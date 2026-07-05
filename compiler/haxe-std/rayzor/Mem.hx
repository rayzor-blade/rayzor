package rayzor;

/**
 * Address-based unchecked memory access. Each function lowers to a single
 * typed load/store MIR instruction (inlined at O2) — no handle deref, no
 * bounds check. The caller owns validity and alignment of `addr`
 * (`Bytes.address()`, `Tensor.data().raw()`, `QTensor.dataPtr()` + offset).
 *
 * `loadF32`/`storeF32` are the only scalar f32 access reachable from Haxe:
 * `Float` is f64, so `Ptr<Float>` reads/writes 8 bytes and corrupts F32
 * buffers; these convert at the boundary instead.
 */
@:native("rayzor::Mem")
extern class Mem {
    /** Load an f32 at `addr`, widened to Float. */
    @:native("loadF32")
    public static function loadF32(addr:Usize):Float;

    /** Narrow `value` to f32 and store at `addr`. */
    @:native("storeF32")
    public static function storeF32(addr:Usize, value:Float):Void;

    /** Load an f64 at `addr`. */
    @:native("loadF64")
    public static function loadF64(addr:Usize):Float;

    /** Store an f64 at `addr`. */
    @:native("storeF64")
    public static function storeF64(addr:Usize, value:Float):Void;

    /** Load a byte at `addr`, zero-extended. */
    @:native("loadU8")
    public static function loadU8(addr:Usize):Int;

    /** Store the low byte of `value` at `addr`. */
    @:native("storeU8")
    public static function storeU8(addr:Usize, value:Int):Void;

    /** Load an i32 at `addr` (aligned). */
    @:native("loadI32")
    public static function loadI32(addr:Usize):Int;

    /** Store an i32 at `addr` (aligned). */
    @:native("storeI32")
    public static function storeI32(addr:Usize, value:Int):Void;
}
