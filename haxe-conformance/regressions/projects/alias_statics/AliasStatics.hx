typedef Mode = abs.Mode;
typedef Flags = abs.Flags;

class AliasStatics {
	static function check(label:String, got:Int, want:Int):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function describe(m:Mode):String
		return switch m {
			case Off: "off";
			case Low: "low";
			case High: "high";
		};

	static function main() {
		var x:Int = Mode.High;
		check("enum abstract value through an alias", x, 2);
		var m:Mode = Mode.Low;
		check("typed local", m, 1);
		if (describe(Mode.High) != "high")
			throw "switch on the alias";
		check("static inline var through an alias", Flags.B, 2);
		check("static var through an alias", Flags.counter, 3);
		check("static method through an alias", Flags.sum(), 6);
		Sys.println("CONFORMANCE_OK");
	}
}
