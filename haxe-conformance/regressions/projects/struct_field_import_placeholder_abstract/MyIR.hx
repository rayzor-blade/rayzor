import pk.cells.Signal;

enum MyRT<T> {
	Const(value:T);
	Bound(state:Signal<T>);
}

abstract MyIR<T>(MyRT<T>) from MyRT<T> to MyRT<T> {
	@:from
	public static inline function fromConst<T>(val:T):MyIR<T> {
		return Const(val);
	}

	@:from
	public static inline function fromState<T>(state:Signal<T>):MyIR<T> {
		return Bound(state);
	}
}
