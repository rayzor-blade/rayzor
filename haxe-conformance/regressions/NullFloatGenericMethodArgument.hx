// A `Null<Float>` or `Null<Bool>` argument for a scalar formal of a generic
// class's method arrives as its value, as it does for a plain function.
class Box<S> {
	public var v = 0.0;
	public var on = false;

	public function new() {}

	public function setWithKind(kind:S, x:Float):Void
		v = x;

	public function flag(kind:S, b:Bool):Void
		on = b;
}

class NullFloatGenericMethodArgument {
	static function main() {
		final d:Null<Float> = 0.3;
		final unset:Null<Float> = null;
		final yes:Null<Bool> = true;
		final b = new Box<Int>();
		b.setWithKind(1, d);
		final first = b.v;
		b.setWithKind(2, unset == null ? 0.5 : unset);
		final second = b.v;
		final s = new Box<String>();
		s.setWithKind("a", d);
		s.flag("b", yes);
		final got = first + " " + second + " " + s.v + " " + s.on;
		if (got != "0.3 0.5 0.3 true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
