// A bare `Middle` takes the expected type's value, an enum abstract of another
// module through `Null<T>`, over a same-named enum constructor or constant.
import toy.draw.GlyphOutlines;

class EnumAbstractValueByExpectedType {
	static function dx(s:TextStyle):Float {
		return switch s.align {
			case Middle: -10.0;
			case End: -20.0;
			case _: 0.0;
		}
	}

	static function dy(s:TextStyle):Float {
		return switch s.baseline {
			case Top: 1.0;
			case Middle: 2.0;
			case Bottom: 3.0;
			case _: 0.0;
		}
	}

	static function main() {
		var s:TextStyle = {size: 24};
		var t:TextStyle = {size: 24, align: Middle, baseline: Bottom};
		var b:Button = Left;
		var got = dx(s) + " " + dy(s) + " " + dx(t) + " " + dy(t) + " " + b;
		if (got != "0 0 -10 3 Left")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
