import haxe.macro.Expr;

using Lambda;

enum Stage { Vertex; Fragment; }

class S {
	public var marked:haxe.EnumFlags<Stage>;
	public var deps:Map<S, Bool>;
	public function new() {
		deps = new Map();
	}
}

class AbstractThisMac {
	static function probe():String {
		var s = new S();
		s.marked = haxe.EnumFlags.ofInt(0);
		var before = s.marked.toInt();
		s.marked.set(Vertex);
		var local = haxe.EnumFlags.ofInt(0);
		local.set(Fragment);
		var arr = [new S(), new S()];
		for (x in arr) x.marked = haxe.EnumFlags.ofInt(0);
		for (x in arr) x.marked.set(Vertex);
		var t = new S();
		s.deps.set(t, true);
		var keys = [for (k in s.deps.keys()) k == t].join("");
		function add3(a:Int, b:Int, c:Int)
			return a * 100 + b * 10 + c;
		var holes = add3.bind(_, 2, _);
		var tail = add3.bind(4);
		var missing = new Map<String, Int>();
		var k:Int = 2;
		var lastK:Int = missing.get("x");
		var nulls = [k <= lastK, k > lastK, k < lastK, k >= lastK].join("");
		var bound = holes(1, 3) + "/" + tail(5, 6) + "/" + [1, 2].has(2) + [1, 2].has(9) + "/" + nulls;
		return bound + "," + before + "," + s.marked.toInt() + "," + s.marked.has(Vertex) + "," + local.toInt() + "," + arr[1].marked.toInt() + "," + keys;
	}

	public static macro function run():Expr
		return macro $v{probe()};
}
