// A Dynamic unifies with an abstract directly: assigning, passing or checking
// it as the abstract keeps the value instead of wrapping it through `@:from`.
enum RT<T> {
	Const(value:T);
	Boxed(label:String);
}

abstract IR<T>(RT<T>) from RT<T> to RT<T> {
	@:from
	public static inline function fromConst<T>(val:T):IR<T>
		return Const(val);
}

class DynamicToAbstractSkipsFrom {
	static function describe<T>(value:IR<T>):String
		return switch (value : RT<T>) {
			case Const(v): "const " + v;
			case Boxed(label): "boxed " + label;
		}

	static function main() {
		var original:IR<Int> = Boxed("x");
		var table:Array<Dynamic> = [original];
		var back:IR<Dynamic> = table[0];
		var checked:IR<Dynamic> = (table[0] : IR<Dynamic>);
		var concrete:IR<Int> = 5;
		var got = [describe(back), describe(table[0]), describe(checked), describe(concrete)].join(",");
		if (got != "boxed x,boxed x,boxed x,const 5")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
