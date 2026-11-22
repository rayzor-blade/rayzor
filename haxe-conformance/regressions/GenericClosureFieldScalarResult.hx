// A generic class's closure field returning Null<T> gives back the scalar
// the closure made, at T = Float and T = Int.
class Tween<T> {
	var from:T;
	var to:T;
	var current:T;
	final lerp:(T, T, Float) -> Null<T>;

	public function new(start:T, lerp:(T, T, Float) -> Null<T>) {
		from = to = current = start;
		this.lerp = lerp;
	}

	public function retarget(value:T):Void {
		from = current;
		to = value;
	}

	public function at(t:Float):Null<T> {
		var v = lerp(from, to, t);
		if (v != null)
			current = v;
		return v;
	}

	public function now():T
		return current;
}

class GenericClosureFieldScalarResult {
	static function main() {
		var f = new Tween<Float>(2.0, (a, b, t) -> a + (b - a) * t);
		f.retarget(10.0);
		var half = f.at(0.5);
		var quarter = f.at(0.25);
		var i = new Tween<Int>(4, (a, b, t) -> a + Std.int((b - a) * t));
		i.retarget(8);
		var got = [Std.string(half), Std.string(quarter), Std.string(f.now()), Std.string(i.at(0.5)), Std.string(i.now() + 1)].join(" ");
		if (got != "6 4 4 6 7")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
