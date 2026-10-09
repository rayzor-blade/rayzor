// Concatenation and interpolation use an abstract's `@:to String` and its own
// `toString`; `this` inlined from an abstract method keeps the underlying type.
private abstract Tagged(Int) from Int {
	@:to function print():String return 'INT $this';
}

private abstract Named(String) from String {
	public function toString() return "named:" + this;
}

private abstract S(String) from String {
	inline function asString():String return this;

	public static function find(b:S) return b.asString().indexOf("ll");
}

class AbstractStringConversion {
	static function main() {
		var t:Tagged = 1;
		var n:Named = "x";
		var got = ["v " + t, 'n $n', "" + S.find("hello")].join(",");
		trace(got == "v INT 1,n named:x,2" ? "CONFORMANCE_OK" : "FAIL " + got);
	}
}
