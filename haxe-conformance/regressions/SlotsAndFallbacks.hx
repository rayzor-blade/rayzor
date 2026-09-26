// Array.shift hands back the element itself for every element type; a `??`
// fallback lambda takes the left side's parameter types; a rest parameter
// behind an omitted optional still receives its array; a function read from a
// Dynamic and stored in a field is callable; a structInit literal fills
// inherited fields; a module-level function returns its body's type; and a
// shift count is taken modulo 32.
private class P {
	public var v:Int;

	public function new(v:Int) this.v = v;
}

@:structInit private class A {
	public var x:Int;
}

@:structInit private class B extends A {
	public var y:Int;
}

private class Holder {
	public var f:Int->Int;

	public function new(d:Dynamic) this.f = d.twice;
}

function plain() return "plain";

class SlotsAndFallbacks {
	static function rest(?first:Int, ...more:Int):Int return more.length;

	static function lookup(?f:Map<Int, String>->String) return f ?? m -> m[1];

	static function main() {
		var out:Array<String> = [];
		var s = ["x", "y"];
		out.push(s.shift());
		var fs = [1.5, 2.5];
		out.push(Std.string(fs.shift()));
		var ps = [new P(7)];
		out.push(Std.string(ps.shift().v));
		out.push(lookup()([1 => "one"]));
		out.push(Std.string(rest()));
		var h = new Holder({twice: (n:Int) -> n * 2});
		out.push(Std.string(h.f(4)));
		var b:B = {x: 1, y: 2};
		out.push(b.x + "," + b.y);
		out.push(plain());
		var n = 33;
		out.push(Std.string(1 << n));
		var got = out.join(" ");
		if (got == "x 1.5 7 one 0 8 1,2 plain 2")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
