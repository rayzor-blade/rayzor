// An enum abstract's values are of the abstract's type, so its own methods apply.
typedef Rec = {
	final a:Int;
	final b:Int;
}

enum abstract E(String) {
	var A = "a";
	var B = "b";

	public function of(set:Rec):Int
		return this == "a" ? set.a : set.b;

	public function pick(set:Rec):Int
		return set.b;
}

enum abstract Level(Int) {
	var Low = 1;
	var High = 5;

	public function scaled(by:Int):Int
		return this * by;
}

class EnumAbstractValueType {
	static function main() {
		var r:Rec = {a: 10, b: 20};
		var inferred = E.B;
		var typed:E = E.A;
		var got = [E.B.of(r), E.A.of(r), inferred.of(r), typed.of(r), E.B.pick(r), High.scaled(2), Level.Low.scaled(3)].join(",");
		if (got != "20,10,20,10,20,10,3")
			throw got;
		if (E.A == E.B || inferred != E.B)
			throw "equality";
		Sys.println("CONFORMANCE_OK");
	}
}
