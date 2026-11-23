package lib;

enum Kind<T> {
	Const(v:T);
	Bound(f:() -> T);
}

abstract Value<T>(Kind<T>) from Kind<T> to Kind<T> {
	@:from public static inline function fromConst<T>(v:T):Value<T>
		return Const(v);
}
