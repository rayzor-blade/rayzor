// A static final regex is one object: a match on one read is visible on the next.
class StaticRegexIdentity {
	static function size(s:String):String {
		var r = Patterns.SIZE;
		return r.match(s) ? r.matched(1) + "|" + r.matched(2) : "none";
	}

	static function main() {
		var same = Patterns.SIZE == Patterns.SIZE;
		Patterns.WORD.match("abc def");
		var carried = Patterns.WORD.matched(0);
		var got = [size("12px"), size("x"), "" + same, carried].join(",");
		if (got != "12|px,none,true,abc")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}

private class Patterns {
	public static final SIZE = ~/^(\d+)([a-z%]*)$/;
	public static final WORD = ~/[a-z]+/;
}
