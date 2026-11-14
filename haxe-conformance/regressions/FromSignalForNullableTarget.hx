// A `Signal<String>` passed for an `Into<Null<String>>` takes the signal
// conversion, not the catch-all constant one.
private interface ISig<T> {
	function get():T;
}

private class StrSig implements ISig<String> {
	var value:String;

	public function new(value:String)
		this.value = value;

	public function get():String
		return value;
}

@:forward(get)
private abstract Sig<T>(ISig<T>) from ISig<T> to ISig<T> {}

private enum Kind<T> {
	Const(value:T);
	Bound(state:Sig<T>);
}

@:forward
private abstract Into<T>(Kind<T>) from Kind<T> to Kind<T> {
	@:from public static inline function fromConst<T>(val:T):Into<T>
		return Const(val);

	@:from public static inline function fromState<T>(state:Sig<T>):Into<T>
		return Bound(state);
}

class FromSignalForNullableTarget {
	static function kind(v:Into<Null<String>>):String
		return switch (v : Kind<Null<String>>) {
			case Const(c): "const:" + c;
			case Bound(s): "bound:" + s.get();
		}

	static function main() {
		var s:Sig<String> = new StrSig("x");
		var got = kind("lit") + " " + kind(s);
		if (got != "const:lit bound:x")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
