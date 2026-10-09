// `overload` functions are picked by argument types: static, member,
// module-level, through `using`, and with optional and rest parameters.
using OverloadCalls.Ext;

class Ext {
	extern overload static public inline function ext(i:Int) return 'Int $i';

	extern overload static public inline function ext(s:String) return 'String $s';
}

private class Statics {
	overload extern static public inline function f(i:Int) return "Int " + i;

	overload extern static public inline function f(s:String) return "String " + s;
}

private class Members {
	public function new() {}

	overload extern public inline function f(i:Int) return "Int " + i;

	overload extern public inline function f(s:String) return "String " + s;
}

private overload extern inline function mod(i:Int) return "Int " + i;

private overload extern inline function mod(s:String) return "String " + s;

class OverloadCalls {
	overload extern static inline function rest(s:String) return "no rest " + s;

	overload extern static inline function rest(s:String, ...r:String) return "rest " + s + " " + r.toArray().join(" ");

	overload extern static inline function opt(?b:Bool, ...a:String) return "opt:" + b + "|" + a.toArray().join(",");

	overload extern static inline function opt(n:Int, ...a:String) return "int:" + n + "|" + a.toArray().join(",");

	static function main() {
		var m = new Members();
		var got = [
			Statics.f(1), Statics.f("a"), m.f(2), m.f("b"), mod(3), mod("c"), 4.ext(), "d".ext(),
			rest("x"), rest("x", "y"), opt(), opt("a"), opt(true, "a"), opt(3, "a", "b")
		].join(";");
		var want = "Int 1;String a;Int 2;String b;Int 3;String c;Int 4;String d;"
			+ "no rest x;rest x y;opt:null|;opt:null|a;opt:true|a;int:3|a,b";
		trace(got == want ? "CONFORMANCE_OK" : "FAIL " + got);
	}
}
