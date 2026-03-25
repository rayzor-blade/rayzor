package rayzor;

/**
 * Read-only reference to a value of type T.
 *
 * Ref<T> is a pointer-sized type with read-only pointer semantics.
 * Passable to C code via `.raw()`. Unlike Ptr<T>, Ref<T> does not
 * allow mutation.
 *
 * Example:
 * ```haxe
 * var ref:Ref<Vec3> = arc.asRef();
 * var v = ref.deref();  // read-only access
 * ```
 */
@:native("rayzor::Ref")
extern abstract Ref<T> {
    /** Create a Ref from a raw address */
    @:native("from_raw")
    public static function fromRaw<T>(address:Usize):Ref<T>;

    /** Get the raw address */
    @:native("raw")
    public function raw():Usize;

    /** Dereference — read the value (read-only) */
    @:native("deref")
    public function deref():T;
}
