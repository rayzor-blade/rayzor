// Methods called on a Dynamic dispatch by the value's own class, String
// included; a Dynamic static's initializer is boxed like an assignment;
// Reflect.callMethod applies defaults for missing arguments and unwraps boxed
// ones; a Dynamic argument unifies with the String a sibling binds.
class Base {
	public function new() {}
	function same<X>(v:X, v2:X):Bool return v == v2;
}

class DynamicCallsAndReflectDefaults extends Base {
	static var empty:Dynamic = [];
	var items:Dynamic = [1, 2];

	function doIt(prefix:String, suffix:Int = 10):String return prefix + suffix;

	static function scaled(x:Float = 1.5, flag:Bool = true):String return x + ":" + flag;

	static var failed = false;

	static function check(label:String, got:Dynamic, want:Dynamic) {
		if (got != want) {
			failed = true;
			Sys.println("FAIL " + label + ": got " + got + " want " + want);
		}
	}

	static function main() {
		var t = new DynamicCallsAndReflectDefaults();
		var s:Dynamic = "str";
		check("array toString", t.items.toString(), "[1,2]");
		check("string toString", s.toString(), "str");
		check("static init", empty.toString() + empty.length, "[]0");
		check("default", Reflect.callMethod(t, t.doIt, ["Alpha"]), "Alpha10");
		check("mixed args", Reflect.callMethod(t, t.doIt, ["Beta", 3]), "Beta3");
		check("static defaults", Reflect.callMethod(null, scaled, [2.5]), "2.5:true");
		var d:Dynamic = "xy";
		check("dynamic unifies", t.same("xy", d), true);
		if (!failed)
			Sys.println("CONFORMANCE_OK");
	}
}
