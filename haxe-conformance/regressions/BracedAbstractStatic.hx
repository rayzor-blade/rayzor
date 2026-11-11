// A braced abstract static whose body the inliner cannot substitute is called.
class V {
	public var n = 7;

	public function new() {}
}

enum R<T> {
	Const(value:T);
	Other(n:Int);
}

abstract Braced<T>(R<T>) from R<T> to R<T> {
	public static inline function make<T>(val:T):Braced<T> {
		return Const(val);
	}

	public static function plain<T>(val:T):Braced<T> {
		return Const(val);
	}
}

abstract Ints(Int) from Int to Int {
	public static function twice(v:Int):Ints {
		return v * 2;
	}

	public static function wrap(v:Int):R<Int> {
		return Const(v);
	}
}

class BracedAbstractStatic {
	static function describe(r:R<V>):String
		return switch r {
			case Const(v): v == null ? "null" : "c" + v.n;
			case Other(n): "o" + n;
		}

	static function main() {
		var v = new V();
		var t:Int = Ints.twice(4);
		var w = switch Ints.wrap(3) {
			case Const(n): "c" + n;
			case Other(_): "o";
		};
		var got = [describe(Braced.make(v)), describe(Braced.plain(v)), "" + t, w].join(",");
		if (got != "c7,c7,8,c3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
