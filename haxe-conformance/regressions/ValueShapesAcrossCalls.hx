// Values keep their representation across calls: Int arithmetic handed to an
// inlined generic still wraps at 32 bits, a virtual call fills its defaults,
// Reflect.setField stores a scalar rather than its box, callMethod unpacks a
// one-parameter function's argument, `Null<Int> - 1` is an Int, `++` on a
// Dynamic field counts, an interface method is a value, and an Int meeting an
// Int64 is 64-bit arithmetic.
private interface Source {
	function read():Float;
}

private class Base {
	public function scale(v:Int = 20):Int return -1;
}

private class Doubler extends Base implements Source {
	public var g:Int;
	public function new() {}
	override public function scale(v:Int = 20):Int return v * 2;
	public function read():Float return 1.5;
}

class ValueShapesAcrossCalls {
	static function show<T>(v:T):String return Std.string(v);

	static function inc(x:Int):Int return x + 1;

	static function same<T>(a:T, b:T):Bool return a == b;

	static function reader(s:Source) return s.read;

	static function main() {
		var out = [];
		var one = Std.parseInt("1"), big = Std.parseInt("2147483647");
		out.push(show(1 << 33) + "," + show(big + one));
		var b:Base = new Doubler();
		out.push(Std.string(b.scale()));
		var d = new Doubler();
		Reflect.setField(d, "g", 6);
		out.push(Std.string(d.g));
		var f = inc;
		out.push(Std.string(Reflect.callMethod(null, f, [5])));
		var n:Null<Int> = 22;
		var m = n - 1;
		out.push(Std.string(same(21, m)));
		var o:Dynamic = {};
		o.count = 1;
		o.count++;
		out.push(Std.string(o.count));
		var r = reader(d);
		out.push(Std.string(r()));
		var wide = haxe.Int64.make(1, 0);
		out.push(Std.string(12345678 - wide == haxe.Int64.make(0xFFFFFFFF, 0x00BC614E)));
		var got = out.join(" ");
		if (got == "2,-2147483648 40 6 6 true 2 1.5 true")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
