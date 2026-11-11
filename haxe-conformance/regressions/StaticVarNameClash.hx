// A static var is its own owner's, whatever other types declare one of the same name.
enum abstract E(Int) to Int {
	var One = 1;
	var Two;

	public static var TAG = "E";
}

abstract Q(Int) {
	public static var TAG = "Q";
}

class P {
	public static var TAG = "P";
}

class StaticVarNameClash {
	static function main() {
		var before = [P.TAG, Q.TAG, E.TAG].join(",");
		E.TAG = "e2";
		var after = [P.TAG, Q.TAG, E.TAG].join(",");
		var v:E = Two;
		var name = switch v {
			case One: "one";
			case Two: "two";
			case _: "?";
		};
		var got = before + ";" + after + ";" + name + (One : Int) + (Two : Int);
		if (got != "P,Q,E;P,Q,e2;two12")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
