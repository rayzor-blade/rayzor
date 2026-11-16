// A class instance read out of a generic enum payload as a Dynamic is the
// instance itself, not its reference box; a type check to the class takes it.
class DynamicInstanceDowncastShadow {
	public final layers:Array<Array<Float>> = [];

	public function new()
		layers.push([1.0, 2.0]);
}

class DynamicInstanceDowncastOther {
	public function new() {}
}

enum DynamicInstanceDowncastCell<T> {
	Const(v:T);
	Other(s:String);
}

class DynamicInstanceDowncast {
	static function read<T>(r:DynamicInstanceDowncastCell<T>):T
		return switch r {
			case Const(v): v;
			case Other(_): null;
		}

	static function takes(r:DynamicInstanceDowncastCell<Dynamic>):Dynamic
		return read(r);

	static function main() {
		var cell:DynamicInstanceDowncastCell<Dynamic> = Const(new DynamicInstanceDowncastShadow());
		var s:DynamicInstanceDowncastShadow = (takes(cell) : DynamicInstanceDowncastShadow);
		var boxed:Dynamic = new DynamicInstanceDowncastShadow();
		var b:DynamicInstanceDowncastShadow = (boxed : DynamicInstanceDowncastShadow);
		var other = Std.downcast(takes(Const(new DynamicInstanceDowncastOther())), DynamicInstanceDowncastShadow);
		var got = s.layers.length + "," + b.layers[0][1] + "," + (other == null);
		if (got != "1,2,true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
