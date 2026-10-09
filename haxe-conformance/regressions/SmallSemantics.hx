// Std.parseInt's null, an assignment in a loop condition, a checked cast
// that throws, `case null` on an enum with arguments, `@:op(a in b)`, and
// a predicate pattern over `_`.
private enum E {
	Bar(i:Int);
}

private enum P {
	CInt(i:Int);
}

private abstract S(String) from String {
	inline function asString() return this;

	@:op(a in b) static function contains(a:S, b:S) return b.asString().indexOf(a.asString()) != -1;
}

class SmallSemantics {
	static function log2(n:Int):Int {
		var res = 0;
		while ((n >>>= 1) != 0)
			res++;
		return res;
	}

	static function range(p:P) {
		return switch p {
			case CInt(_ > 0 && _ < 12): "in";
			case _: "out";
		}
	}

	static function main() {
		var parsed:Null<Int> = Std.parseInt("axolotl");
		var threw = try {
			var x:Dynamic = "foo";
			cast(x, Int);
			false;
		} catch (e:Dynamic) true;
		var nullCase = switch Bar(4) {
			case null: "null";
			case Bar(v): "bar " + v;
		}
		var s:S = "hello";
		var got = [
			"" + (parsed == null), "" + log2(16), "" + threw, nullCase, "" + ("hell" in s), range(CInt(11)), range(CInt(12))
		].join(",");
		var want = "true,4,true,bar 4,true,in,out";
		trace(got == want ? "CONFORMANCE_OK" : "FAIL " + got);
	}
}
