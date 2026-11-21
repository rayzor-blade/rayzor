import pk.Cell;

enum Value<T> {
	Const(value:T);
	Bound(cell:Cell<T>);
}

abstract Reactive<T>(Value<T>) from Value<T> to Value<T> {
	@:from
	public static inline function fromConst<T>(val:T):Reactive<T> {
		return Const(val);
	}

	@:from
	public static inline function fromCell<T>(cell:Cell<T>):Reactive<T> {
		return Bound(cell);
	}
}
