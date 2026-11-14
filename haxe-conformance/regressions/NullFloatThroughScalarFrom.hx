enum RType<T> {
	Const(value:T);
}

abstract IntoR<T>(RType<T>) from RType<T> to RType<T> {
	@:from public static inline function fromConst<T>(val:T):IntoR<T>
		return Const(val);

	@:from public static inline function fromInt(val:Int):IntoR<Single>
		return Const((val : Single));

	@:from public static inline function fromFloat(val:Float):IntoR<Single>
		return Const((val : Single));
}

typedef Attrs = {?width:IntoR<Single>, ?height:IntoR<Single>}

// A Null<Float> for an IntoR<Single> converts through fromFloat, as Float
// takes a Null<Float>; fromConst would infer IntoR<Null<Float>>.
class NullFloatThroughScalarFrom {
	static function read(a:Attrs):String {
		return switch (a.width : RType<Single>) {
			case Const(v): Std.string(v);
		}
	}

	static function main() {
		var p:{?height:Float} = {height: 200};
		var size = p.height != null ? p.height : 220.0;
		var q:{?height:Float} = {};
		var other = q.height != null ? q.height : 220.0;
		var got = read({width: size, height: size}) + " " + read({width: other});
		if (got != "200 220")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
