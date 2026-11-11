// `$a{list}` beside other call arguments is one array; alone it is the argument list.
class MacroArgSplice {
	public static function sum(base:Int, values:Array<Int>):Int {
		var t = base;
		for (v in values)
			t += v;
		return t;
	}

	public static function three(a:Int, b:Int, c:Int):Int
		return a + b * 10 + c * 100;

	public static function count(radial:Bool, stops:Array<{offset:Float, alpha:Float}>):Int
		return stops.length;

	static function main() {
		var got = [SpliceMac.beside(), SpliceMac.alone(), SpliceMac.objects()].join(",");
		if (got != "61,321,3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
