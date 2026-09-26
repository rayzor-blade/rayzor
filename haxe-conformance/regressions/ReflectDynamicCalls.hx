// Reflect.callMethod calls a function held as Dynamic with the array's
// elements; a method read by a literal name through Reflect.field is a bound
// closure; a @:native method is found under its native name, overrides
// included; Reflect.field reads a String's length.
private class Counter {
	var base:Int;

	public function new(base:Int) this.base = base;

	public function add(a:Int, b:Int):Int return base + a + b;

	public function get():Int return base;
}

private class Base {
	public function new() {}

	@:native("renamed") public function original():String return "B";
}

private class Derived extends Base {
	override public function original():String return "D";
}

class ReflectDynamicCalls {
	static function main() {
		var out:Array<String> = [];
		var c = new Counter(10);
		var get:Dynamic = Reflect.field(c, "get");
		out.push(Std.string(Reflect.callMethod(c, get, [])));
		var add:Dynamic = Reflect.field(c, "add");
		out.push(Std.string(Reflect.callMethod(c, add, [1, 2])));
		var lambda:Dynamic = (x:Int) -> x * 3;
		out.push(Std.string(Reflect.callMethod(null, lambda, [5])));
		var d = new Derived();
		var renamed:Dynamic = Reflect.field(d, "renamed");
		out.push(Std.string(renamed != null));
		out.push(Std.string(Reflect.callMethod(d, renamed, [])));
		var s = "hello";
		out.push(Std.string(Reflect.field(s, "length")));
		var got = out.join(" ");
		if (got == "10 13 15 true D 5")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
