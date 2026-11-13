// A generic `@:from` taking `Holder<T>` converts any `Holder<X>` bound to an
// abstract over an enum, not only the exact declared instantiation.
class FromHolder<T> {
	public var v:T;

	public function new(v:T)
		this.v = v;
}

enum FromSource<T> {
	Literal(value:T);
	Held(state:FromHolder<T>);
}

abstract FromInto<T>(FromSource<T>) from FromSource<T> to FromSource<T> {
	@:from public static inline function fromConst<T>(val:T):FromInto<T>
		return Literal(val);

	@:from public static inline function fromHolder<T>(state:FromHolder<T>):FromInto<T>
		return Held(state);
}

class GenericFromOverGenericClass {
	static function main() {
		var a:FromInto<Float> = new FromHolder<Float>(0.9);
		var h = new FromHolder<Float>(0.5);
		var b:FromInto<Float> = h;
		var c:FromInto<Float> = 0.25;
		var got = [Type.enumConstructor((a : FromSource<Float>)), Type.enumConstructor((b : FromSource<Float>)),
			Type.enumConstructor((c : FromSource<Float>))].join(",");
		if (got != "Held,Held,Literal")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
