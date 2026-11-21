package lib.ui;

enum Value<T> {
	Const(value:T);
	Bound(label:String);
}

abstract Reactive<T>(Value<T>) from Value<T> to Value<T> {
	@:from
	public static inline function fromConst<T>(val:T):Reactive<T> {
		return Const(val);
	}
}
