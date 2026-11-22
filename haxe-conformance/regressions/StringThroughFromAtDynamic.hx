// A String converted by a generic `@:from` at a concrete expected
// abstract over Dynamic arrives as a String Dynamic.
enum StrReactive<T> {
	Fixed(value:T);
	Lazy(get:Void->T);
}

abstract IntoStr<T>(StrReactive<T>) from StrReactive<T> to StrReactive<T> {
	@:from public static inline function fromValue<T>(value:T):IntoStr<T>
		return Fixed(value);

	@:from public static inline function fromLazy<T>(get:Void->T):IntoStr<T>
		return Lazy(get);
}

class StringThroughFromAtDynamic {
	static function read(value:IntoStr<Dynamic>):Dynamic
		return switch (value : StrReactive<Dynamic>) {
			case Fixed(v): v;
			case Lazy(get): get();
		};

	static function describe(value:IntoStr<Dynamic>):String {
		var v:Dynamic = read(value);
		return Std.string(v) + ":" + Std.isOfType(v, String) + ":" + (v : String).length;
	}

	static function main() {
		var got = [describe("h"), describe("word"), describe(Lazy(() -> ("lazy" : Dynamic)))].join(" ");
		if (got != "h:true:1 word:true:4 lazy:true:4")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
