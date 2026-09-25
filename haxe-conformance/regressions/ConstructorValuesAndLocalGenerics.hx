// Constructors used as values, calls through local generic functions, a
// type parameter only Dynamic arguments reach, and lambdas a bare `return;`
// makes Void. This file begins with a byte order mark the lexer skips.
private abstract Stack<T>(Int) {
	public function new(i) this = i;
	public function get() return this;
}

private abstract Name(String) to String {
	public function new(s:String) this = s;
	public inline function get() return this;
}

private class Point {
	public var x:Int;
	public function new(x:Int) this.x = x;
}

class ConstructorValuesAndLocalGenerics {
	function new() {}

	function same<T>(a:Array<T>, b:Array<T>):Bool {
		if (a.length != b.length) return false;
		for (i in 0...a.length)
			if (a[i] != b[i] && Std.string(a[i]) != Std.string(b[i])) return false;
		return true;
	}

	function is<T>(a:T, b:T):Bool return a == b;

	static function run(f:Bool->Void) f(true);

	function check():String {
		var p = Point.new;
		var s = Stack.new;
		var n = Name.new.bind("hey");
		var built = p(4).x + s(7).get();
		function wrap<T>(v:T):Array<T> return [v];
		function id<T>(v:T):T return v;
		var generic = same([3], wrap(3)) && is(5, id(5));
		var dynamic = same(([1, "a"]:Array<Dynamic>), rest(1, "a"));
		var calls = 0;
		var early = b -> if (b) calls++ else return;
		run(early);
		var viaSwitch = b -> switch b {
			case true: calls += 10;
			case false: return;
		}
		run(viaSwitch);
		return '$built ${n().get()} $generic $dynamic $calls';
	}

	function rest(...args:Dynamic) return args.toArray();

	static function main() {
		var got = new ConstructorValuesAndLocalGenerics().check();
		if (got == "11 hey true true 11")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
