// A switch expression with Int and Float arms is a Float, whichever comes first.
class SwitchMixedNumericArms {
	static function width(part:String, f:Float):Float {
		return switch part {
			case "thin": 1;
			case "thick": 5;
			case _: Math.max(0, f);
		}
	}

	static function main() {
		var zero = "".length;
		var inferred = switch zero {
			case 0: 2;
			case _: 0.5;
		}
		var typed:Float = switch "x" {
			case "a": 1;
			case _: 2.25;
		}
		var got = width("x", 1.5) + "," + width("thin", 1.5) + "," + (inferred / 4) + "," + typed;
		if (got != "1.5,1,0.5,2.25")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
