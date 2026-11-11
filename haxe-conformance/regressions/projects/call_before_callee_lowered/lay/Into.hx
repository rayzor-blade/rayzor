package lay;

enum Kind<T> {
	Const(value:T);
	Missing;
}

abstract Into<T>(Kind<T>) from Kind<T> to Kind<T> {
	@:from
	public static inline function fromConst<T>(val:T):Into<T> {
		return Const(val);
	}
}
