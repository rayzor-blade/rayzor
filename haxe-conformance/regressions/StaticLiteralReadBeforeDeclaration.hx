// An unannotated static array or regex, read by a method declared above its
// class, has its literal's type.
class StaticLiteralReadBeforeDeclaration {
	static function main() {
		var got = Reader.count() + "|" + Reader.total() + "|" + Reader.match("12px");
		if (got != "3|3.5|12")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}

class Reader {
	public static function count():Int
		return Store.items.length;

	public static function total():Float {
		var sum = 0.0;
		for (v in Store.mixed)
			sum += v;
		return sum;
	}

	public static function match(s:String):String
		return Store.RE.match(s) ? Store.RE.matched(1) : "none";
}

class Store {
	public static final items = [1, 2, 3];
	public static final mixed = [1, 2.5];
	public static final RE = ~/^(\d+)px$/;
}
