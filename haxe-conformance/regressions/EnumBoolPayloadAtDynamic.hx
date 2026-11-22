// A Bool in a generic enum's payload reads back as a Bool at T = Dynamic.
enum Held<T> {
	Fixed(x:T);
	Lazy(get:Void->T);
}

class EnumBoolPayloadAtDynamic {
	static function wrap<T>(v:T):Held<T>
		return Fixed(v);

	static function read(h:Held<Dynamic>):Dynamic
		return switch h {
			case Fixed(x): x;
			case Lazy(get): get();
		};

	static function main() {
		var typed:Bool = true;
		var off:Dynamic = false;
		var fromTyped:Held<Dynamic> = Fixed(typed);
		var values:Array<Dynamic> = [
			read(Fixed(true)),
			read(wrap((false : Dynamic))),
			read(fromTyped),
			read(Fixed(off)),
			read(Fixed(7)),
		];
		var got = [for (v in values) Std.string(v) + ":" + Std.isOfType(v, Bool)].join(" ");
		var same = read(Fixed(true)) == true;
		if (got != "true:true false:true true:true false:true 7:false" || !same)
			throw got + " " + same;
		Sys.println("CONFORMANCE_OK");
	}
}
