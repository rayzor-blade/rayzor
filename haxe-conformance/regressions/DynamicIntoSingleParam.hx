// A Dynamic value passed to a Single parameter is the number it holds.
class DynamicIntoSingleParam {
	static function take(x:Single, y:Single):String {
		return x + "," + y;
	}

	static inline function inlined(x:Single):Float {
		return x * 2;
	}

	static function main() {
		var s:Dynamic = {x: 3.0, y: 7.5, z: 0.0};
		var got = take(s.x, s.y) + "|" + inlined(s.y) + "|" + take(s.z, s.z);
		if (got != "3,7.5|15|0,0")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
