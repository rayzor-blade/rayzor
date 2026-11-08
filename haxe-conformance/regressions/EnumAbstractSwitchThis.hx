enum abstract Id(Int) from Int to Int {
	var A = 1;
	var B = 2;
	var C = 3;
	var D = 4;

	public inline function inlineAlt():Int {
		return switch (this) {
			case A | B: 10;
			case C: 30;
			case D: 40;
			case _: 0;
		}
	}

	public function plainSingle():Int {
		return switch (this) {
			case A: 10;
			case B: 20;
			case C: 30;
			case D: 40;
			case _: 0;
		}
	}

	public static function fromStatic(v:Id):Int {
		return switch (v) {
			case A | B: 10;
			case C: 30;
			case D: 40;
			case _: 0;
		}
	}
}

class EnumAbstractSwitchThis {
	static function main() {
		var a:Id = Id.A;
		var c:Id = Id.C;
		var d:Id = Id.D;
		var other:Id = 9;
		if (a.inlineAlt() != 10 || c.inlineAlt() != 30 || d.inlineAlt() != 40 || other.inlineAlt() != 0) throw "inline alternatives";
		if (a.plainSingle() != 10 || c.plainSingle() != 30 || d.plainSingle() != 40 || other.plainSingle() != 0) throw "plain single";
		if (Id.fromStatic(c) != 30 || Id.fromStatic(d) != 40) throw "static";
		Sys.println("CONFORMANCE_OK");
	}
}
