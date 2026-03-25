package rayzor;

/**
 * Unsigned pointer-sized integer.
 *
 * Usize is a pointer-sized unsigned integer — the natural type for
 * memory addresses, sizes, offsets, and array indices.
 *
 * On rayzor's baremetal target, Usize is always 64-bit. It is a
 * first-class type, NOT an abstract over Int (which is 32-bit).
 *
 * Convertible from/to Int via fromInt()/toInt(). Use `.toPtr()` and
 * `.toRef()` for typed pointer conversions.
 *
 * Example:
 * ```haxe
 * var addr:Usize = 0x1000;           // implicit @:from Int
 * var ptr:Ptr<Vec3> = addr.toPtr();
 * var size:Int = addr;               // implicit @:to Int
 * ```
 */
@:native("rayzor::Usize")
extern abstract Usize {
    /** Create from an Int value */
    @:native("from_int")
    @:from
    public static function fromInt(value:Int):Usize;

    /** Convert to Int */
    @:native("to_int")
    @:to
    public function toInt():Int;

    /** Create from a raw pointer address */
    @:native("from_ptr")
    public static function fromPtr<T>(ptr:Ptr<T>):Usize;

    /** Create from a raw ref address */
    @:native("from_ref")
    public static function fromRef<T>(ref:Ref<T>):Usize;

    /** Convert to a typed mutable pointer */
    @:native("to_ptr")
    public function toPtr<T>():Ptr<T>;

    /** Convert to a typed read-only reference */
    @:native("to_ref")
    public function toRef<T>():Ref<T>;

    /** Add an offset (pointer arithmetic) */
    @:native("add")
    @:op(A + B)
    public function add(other:Usize):Usize;

    /** Subtract an offset */
    @:native("sub")
    @:op(A - B)
    public function sub(other:Usize):Usize;

    /** Bitwise AND */
    @:native("band")
    @:op(A & B)
    public function band(other:Usize):Usize;

    /** Bitwise OR */
    @:native("bor")
    @:op(A | B)
    public function bor(other:Usize):Usize;

    /** Left shift */
    @:native("shl")
    @:op(A << B)
    public function shl(bits:Int):Usize;

    /** Right shift (unsigned) */
    @:native("shr")
    @:op(A >>> B)
    public function shr(bits:Int):Usize;

    /** Align up to the given alignment (must be power of 2) */
    @:native("align_up")
    public function alignUp(alignment:Usize):Usize;

    /** Check if value is zero */
    @:native("is_zero")
    public function isZero():Bool;
}
