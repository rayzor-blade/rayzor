// Values convert where Haxe converts them: a `case var x` capture has the
// subject's type, an Int passed beside a Float-bound T is a Float, an Int
// literal fills a Float field, a literal typed as an abstract over a
// @:structInit class builds that class, a constructor argument goes through
// @:from, a Null<abstract> unboxes into the abstract, an inlined abstract
// method calls Array methods on its underlying value, and malformed JSON
// throws.
private enum abstract Level(Int) {
	var Low = 0;
	var High = 1;
}

private abstract Stack(Array<Int>) {
	public function new() this = [12];

	public function top() return this.pop();
}

@:structInit private class PointImpl {
	public var x:Int;
}

@:forward private abstract Point(PointImpl) from PointImpl to PointImpl {}

private class Box {
	public var i:Int;

	public function new(i:Int) this.i = i;
}

private abstract Wrapped(Box) from Box to Box {
	@:from static function fromArr(a:Array<Int>):Wrapped return new Box(a[0]);
}

private class Holder {
	public var w:Wrapped;

	public function new(w:Wrapped) this.w = w;
}

typedef Measure = {f:Float};

class CoercionPoints {
	static function same<T>(a:T, b:T):Bool return a == b;

	static function main() {
		var out = [];
		var o:Null<Int> = 0;
		out.push(switch o {
			case var x if (x > 1): "wrong";
			case _: "right";
		});
		out.push(Std.string(same(1.0, 1)));
		var m:Measure = {f: -1};
		out.push(Std.string(0.5 > m.f));
		var p:Point = {x: 2};
		out.push(Std.string(p.x));
		var h = new Holder([7]);
		var b:Box = h.w;
		out.push(Std.string(b.i));
		var n:Null<Level> = High;
		var l:Level = n;
		out.push(Std.string(l == High));
		out.push(Std.string(new Stack().top()));
		var threw = try {
			haxe.Json.parse('{"a":}');
			false;
		} catch (e:Dynamic) true;
		out.push(Std.string(threw));
		var got = out.join(" ");
		if (got == "right true true 2 7 true 12 true")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
