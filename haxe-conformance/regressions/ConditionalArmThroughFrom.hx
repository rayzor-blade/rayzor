// A conditional typed as an abstract converts each arm on its own: an arm
// already of the abstract passes, the other goes through its @:from.
#if rayzor
typedef ArmNumber = Single;
#else
typedef ArmNumber = Float;
#end

enum ArmSource<T> {
	ArmConstant(value:T);
}

abstract ArmInto<T>(ArmSource<T>) from ArmSource<T> to ArmSource<T> {
	@:from public static inline function fromConst<T>(val:T):ArmInto<T>
		return ArmConstant(val);
}

typedef ArmAttr = {?width:ArmInto<ArmNumber>, ?height:ArmInto<ArmNumber>};

class ConditionalArmThroughFrom {
	static function read<T>(value:ArmInto<T>):String
		return switch (value : ArmSource<T>) {
			case ArmConstant(v): Std.string(v);
		}

	static function pass(fallback:Int, ?attr:ArmAttr):String
		return read(attr != null && attr.width != null ? attr.width : (fallback : ArmNumber));

	static function local(fallback:Int, ?attr:ArmAttr):String {
		var w:ArmInto<ArmNumber> = attr != null && attr.width != null ? attr.width : (fallback : ArmNumber);
		return read(w);
	}

	static function main() {
		var given:ArmAttr = {width: (50 : ArmNumber)};
		var got = [pass(24), pass(24, {}), pass(24, given), local(24), local(24, given)].join(",");
		if (got != "24,24,50,24,50")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
