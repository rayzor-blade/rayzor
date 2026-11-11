// A conditional with an Int branch and a Float branch is a Float, as Haxe unifies them.
class ConditionalIntFloat {
	static function sink(f:Float):Float
		return f;

	static function main() {
		var none = false;
		var width = 4.5;
		var i = 3;
		var a = none ? 0 : width;
		var b = none ? i : width;
		var c = none ? 1 : 2.5;
		var d = if (none) 0 else width;
		var w:Float = none ? 0 : width;
		var singles:Array<Single> = [none ? 0 : width];
		var n = 0;
		var t = 0.0;
		if (none) n = 1 else t = 0.5;
		if (!none) n += 2 else t += 1.5;
		var got = [a, b, c, d, w, sink(none ? 0 : width), singles[0], none ? width : 0].join(",") + "|" + n + "," + t;
		if (got != "4.5,4.5,2.5,4.5,4.5,4.5,4.5,0|2,0.5")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
