// `case A | B` over an enum abstract value matches those values only.
enum abstract Id(Int) from Int to Int {
	var A = 1;
	var B = 2;
	var C = 3;

	public static function inside(v:Id):Int
		return switch (v) {
			case A | B: 10;
			case C: 30;
			case _: 0;
		}
}

class EnumAbstractOrPattern {
	static function bar(v:Id):Int
		return switch (v) {
			case A | B: 10;
			case C: 30;
			case _: 0;
		}

	static function comma(v:Id):Int
		return switch (v) {
			case A, B: 10;
			case C: 30;
			case _: 0;
		}

	static function main() {
		var got = [bar(Id.A), bar(Id.B), bar(Id.C), bar(9), comma(Id.C), comma(Id.B), Id.inside(Id.C), Id.inside(Id.A)].join(",");
		if (got != "10,10,30,0,30,10,30,10")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
