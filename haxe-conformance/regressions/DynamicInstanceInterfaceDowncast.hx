// A class instance read out of a generic enum payload as a Dynamic and
// checked to an interface it implements is that interface's value for it.
interface DynamicInstanceInterfaceDowncastValue {
	public var ptr(default, null):Int;
}

class DynamicInstanceInterfaceDowncastRadius implements DynamicInstanceInterfaceDowncastValue {
	public var ptr(default, null):Int = 7;
	public final layers:Array<Array<Float>> = [];

	public function new()
		layers.push([1.0, 2.0]);
}

class DynamicInstanceInterfaceDowncastOther {
	public function new() {}
}

enum DynamicInstanceInterfaceDowncastCell<T> {
	Const(v:T);
	Other(s:String);
}

class DynamicInstanceInterfaceDowncast {
	static function read<T>(r:DynamicInstanceInterfaceDowncastCell<T>):T
		return switch r {
			case Const(v): v;
			case Other(_): null;
		}

	static function takes(r:DynamicInstanceInterfaceDowncastCell<Dynamic>):Dynamic
		return read(r);

	static function mix(a:DynamicInstanceInterfaceDowncastValue, b:DynamicInstanceInterfaceDowncastValue):String {
		if (!Std.isOfType(a, DynamicInstanceInterfaceDowncastRadius) || !Std.isOfType(b, DynamicInstanceInterfaceDowncastRadius))
			return "null";
		return "value";
	}

	static function main() {
		var cell:DynamicInstanceInterfaceDowncastCell<Dynamic> = Const(new DynamicInstanceInterfaceDowncastRadius());
		var p:DynamicInstanceInterfaceDowncastValue = (takes(cell) : DynamicInstanceInterfaceDowncastValue);
		var back:DynamicInstanceInterfaceDowncastRadius = cast p;
		var other = Std.downcast(takes(Const(new DynamicInstanceInterfaceDowncastOther())), DynamicInstanceInterfaceDowncastValue);
		var got = p.ptr + "," + Std.isOfType(p, DynamicInstanceInterfaceDowncastRadius) + "," + back.layers.length + "," + mix(p, p) + ","
			+ (other == null);
		if (got != "7,true,1,value,true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
