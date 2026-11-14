// `(d : Single)` of a Dynamic holding a Single is that Single, as a tween's
// start value read back from a Dynamic slot is.
#if rayzor
typedef DtNumber = Single;
#else
typedef DtNumber = Float;
#end

class DynamicTypeCheckedAsSingle {
	static function initial(raw:Int):Null<Dynamic> {
		return switch raw {
			case 4: (1.0 : DtNumber);
			case _: null;
		}
	}

	static function readV(v:Dynamic):Dynamic
		return v;

	static function start(value:Dynamic, ?from:Dynamic):DtNumber {
		inline function read(value:Dynamic):Dynamic
			return from != null ? from : readV(value);
		return (read(value) : DtNumber);
	}

	static function main() {
		var target:DtNumber = 0.5;
		var d:Dynamic = initial(4);
		var mixed = (d : DtNumber) + ((target : DtNumber) - (d : DtNumber)) * 0.5;
		var got = [start(target, initial(4)), start(target), mixed].map(v -> Math.round(v * 100)).join(",");
		if (got != "100,50,75")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
