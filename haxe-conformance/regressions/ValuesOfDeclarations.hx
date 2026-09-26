// An enum constructor is a function value; a class stored in an
// interface-typed field keeps its interface view; a Null<Int> static holds a
// box and `/` on two of them is Float division; `final m:Map = []` is a map;
// a loop over an abstract uses its iterator(); a block-bodied arrow function
// returns its last expression.
private enum Shape {
	Circle(r:Int);
	Pair(a:Int, b:String);
}

private interface Named {
	function name():String;
}

private class Tag implements Named {
	public function new() {}

	public function name():String return "tag";
}

private class Holder {
	public var named:Named;

	public function new() {}
}

private abstract Letters(String) from String {
	public function iterator():Iterator<String> {
		var i = 0;
		var s = this;
		return {hasNext: () -> i < s.length, next: () -> s.charAt(i++)};
	}
}

class ValuesOfDeclarations {
	static var half:Null<Int> = 1;
	static var whole:Null<Int> = 2;

	static function describe(s:Shape):String {
		return switch (s) {
			case Circle(r): "c" + r;
			case Pair(a, b): "p" + a + b;
		}
	}

	static function main() {
		var out:Array<String> = [];
		var make = Circle;
		out.push(describe(make(3)));
		var shapes = [1, 2].map(Circle);
		out.push(describe(shapes[1]));
		var pair:(Int, String) -> Shape = Pair;
		out.push(describe(pair(4, "x")));
		var h = new Holder();
		h.named = new Tag();
		out.push(h.named.name());
		out.push(Std.string(half / whole));
		final m:Map<Int, Int> = [];
		m[3] = 9;
		out.push(Std.string(m[3]));
		var letters:Letters = "ab";
		var joined = "";
		for (c in letters)
			joined += c;
		out.push(joined);
		var tail = () -> {
			var x = 2;
			"t" + x;
		};
		out.push(tail());
		var got = out.join(" ");
		if (got == "c3 c2 p4x tag 0.5 9 ab t2")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
