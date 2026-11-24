// A value handed to a helper is released after the call only when the helper
// keeps no reference to it; every way of keeping one leaves it alive.
class Box {
	public var s:String;
	public var other:String;

	public function new() {
		s = "";
		other = "";
	}
}

class HelperArgumentRetention {
	static var saved:String = "";
	static final byKey = new Map<String, Int>();
	static var later:Void->String;

	static function measure(s:String):Int
		return s.length + s.charCodeAt(0);

	static function pick(b:Box, s:String, flag:Bool):Void {
		var chosen = flag ? s : b.other;
		b.s = chosen;
	}

	static function remember(s:String):Void
		saved = s;

	static function index(s:String, v:Int):Void
		byKey.set(s, v);

	static function defer(s:String):Void
		later = () -> s + "!";

	static function same(s:String):String
		return s;

	static function main() {
		var total = 0;
		var b = new Box();
		for (i in 0...4) {
			total += measure("m" + i);
			pick(b, "p" + i, i % 2 == 1);
			remember("r" + i);
			index("k" + i, i);
			defer("d" + i);
		}
		var kept = same("s" + total);
		var keys = [for (k in byKey.keys()) k];
		keys.sort((x, y) -> x < y ? -1 : (x > y ? 1 : 0));
		final got = total + "," + b.s + "," + saved + "," + keys.join("") + "," + later() + "," + kept;
		final want = "444,p3,r3,k0k1k2k3,d3!,s444";
		if (got != want)
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
