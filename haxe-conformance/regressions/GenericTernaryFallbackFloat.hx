// A generic ternary whose then-arm is a `T` parameter and whose else-arm is a
// switch expression keeps the value when `T` is bound to Float.
enum GenTernaryValue<T> {
	Some(v:T);
	Empty;
}

class GenericTernaryFallbackFloat {
	static function pick<T>(e:Null<GenTernaryValue<T>>, fallback:T):T
		return e == null ? fallback : switch e {
			case Some(v): v;
			case Empty: fallback;
		}

	static function main() {
		var got = [Std.string(pick(Some(0.4), 0.25)), Std.string(pick(Empty, 0.25)), Std.string(pick(null, 0.25)),
			Std.string(pick(Some(7), 1)), pick(Some("s"), "d")].join(",");
		if (got != "0.4,0.25,0.25,7,s")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
