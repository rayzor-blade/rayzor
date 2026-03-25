package rayzor;

/**
 * Raw mutable pointer to a value of type T.
 *
 * Ptr<T> is a pointer-sized type that provides typed pointer semantics.
 * Passable to C code via `.raw()`.
 *
 * Ptr, Ref, Box, and Usize are first-class 64-bit types on rayzor's
 * baremetal target. They do NOT abstract over Int (which is 32-bit).
 *
 * With `@:cstruct` classes, the memory layout matches C exactly,
 * so pointers are directly interoperable.
 *
 * Example:
 * ```haxe
 * var ptr:Ptr<Vec3> = Ptr.fromRaw(address);
 * var v = ptr.deref();
 * ptr.write(newValue);
 * ```
 */
@:native("rayzor::Ptr")
extern abstract Ptr<T> {
    /** Create a Ptr from a raw address */
    @:native("from_raw")
    public static function fromRaw<T>(address:Usize):Ptr<T>;

    /** Get the raw address */
    @:native("raw")
    public function raw():Usize;

    /** Dereference — read the value at this pointer */
    @:native("deref")
    public function deref():T;

    /** Write a value at this pointer */
    @:native("write")
    public function write(value:T):Void;

    /** Pointer arithmetic — offset by N elements of size T */
    @:native("offset")
    public function offset(n:Int):Ptr<T>;

    /** Check if this pointer is null */
    @:native("isNull")
    public function isNull():Bool;
}
