// At an expected `A<Dynamic>`, a `@:from` over `Signal<T>` takes a
// `Signal<Dynamic>` before a catch-all `from(val:T)`; a `Signal<String>`,
// which is not a `Signal<Dynamic>`, takes the catch-all.
class Signal<T> {
	var v:T;

	public function new(v:T)
		this.v = v;

	public function get():T
		return v;
}

class Computed<T> {
	final f:Void->T;

	public function new(f:Void->T)
		this.f = f;

	public function get():T
		return f();
}

enum ReactiveType<T> {
	Const(value:T);
	Bound(state:Signal<T>);
	Derived(computed:Computed<T>);
}

abstract IntoReactive<T>(ReactiveType<T>) from ReactiveType<T> to ReactiveType<T> {
	@:from public static inline function fromConst<T>(val:T):IntoReactive<T>
		return Const(val);

	@:from public static inline function fromState<T>(state:Signal<T>):IntoReactive<T>
		return Bound(state);

	@:from public static inline function fromComputed<T>(comp:Computed<T>):IntoReactive<T>
		return Derived(comp);
}

class FromSelectionAtDynamicExpected {
	static function kind(value:IntoReactive<Dynamic>):String
		return switch (value : ReactiveType<Dynamic>) {
			case Const(_): "const";
			case Bound(_): "bound";
			case Derived(_): "derived";
		}

	static function main() {
		var got = [
			kind("a"),
			kind(new Signal<Dynamic>("b")),
			kind(new Signal<String>("c")),
			kind(new Computed<Dynamic>(() -> "d")),
			kind(new Computed<String>(() -> "e")),
		].join(" ");
		if (got != "const bound const derived const")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
