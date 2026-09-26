// A method called or read through a Dynamic or a structurally typed value is
// found by name at run time -- a class instance's method, or an anonymous
// object's function field -- never bound to an unrelated method that shares
// its name.
typedef Fetcher = {function fetch():Int;}

typedef Getter = {get:() -> Int};

private class Counter {
	var base:Int;

	public function new(base:Int) this.base = base;

	public function fetch():Int return base + 1;

	public function add(a:Int, b:Int):Int return a + b + base;
}

class DynamicMethodDispatch {
	static function main() {
		var out:Array<String> = [];
		var d:Dynamic = new Counter(10);
		out.push(Std.string(d.fetch()));
		out.push(Std.string(d.add(1, 2)));
		var e:Dynamic = {fetch: () -> 7, add: (a:Int, b:Int) -> a * b};
		out.push(Std.string(e.fetch()));
		out.push(Std.string(e.add(3, 4)));
		var viaDynamic:Dynamic = d.fetch;
		out.push(Std.string(viaDynamic()));
		var f:Fetcher = new Counter(20);
		out.push(Std.string(f.fetch()));
		var read = f.fetch;
		out.push(Std.string(read()));
		var g:Getter = {get: () -> 5};
		out.push(Std.string(g.get()));
		var got = out.join(" ");
		if (got == "11 13 7 12 11 21 21 5")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
