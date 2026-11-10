interface IVal {
	var ptr(default, null):Int;
	function name():String;
}

class Br implements IVal {
	public var ptr(default, null):Int;

	public function new(p:Int)
		ptr = p;

	public function name():String
		return "brush";
}

enum RT<T> {
	Const(v:T);
	Other(n:Int);
}

abstract IR<T>(RT<T>) from RT<T> to RT<T> {
	@:from static function fromConst<T>(v:T):IR<T>
		return Const(v);
}

class Cell<T> {
	var value:T;

	public function new(v:T)
		value = v;

	public function put(v:T):Void
		value = v;

	public function get():T
		return value;
}

class InstantiationRepresentations {
	static function ptrOf(r:IR<IVal>):Int
		return switch (r : RT<IVal>) {
			case Const(v): v.ptr;
			case Other(n): n;
		};

	static function nameOf(r:IR<IVal>):String
		return switch (r : RT<IVal>) {
			case Const(v): v.name();
			case Other(_): "other";
		};

	static function viaGeneric<T>(r:IR<T>):Int
		return ptrOf(cast r);

	static function putFloat(c:Cell<Dynamic>, v:Float):Void
		c.put(v);

	static function putBool(c:Cell<Dynamic>, v:Bool):Void
		c.put(v);

	static function putString(c:Cell<Dynamic>, v:String):Void
		c.put(v);

	static function putInt(c:Cell<Dynamic>, v:Int):Void
		c.put(v);

	static function check(label:String, got:String, want:String):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		var declared:IR<IVal> = (new Br(77) : IVal);
		check("declared", Std.string(ptrOf(declared)), "77");
		var r:IR<Br> = new Br(77);
		check("cast between instantiations", Std.string(ptrOf(cast r)), "77");
		check("cast inside a generic", Std.string(viaGeneric(r)), "77");
		check("method after the cast", nameOf(cast r), "brush");

		var c = new Cell<Dynamic>(null);
		putFloat(c, 0.5);
		check("Float into Cell<Dynamic>", Std.string(c.get()), "0.5");
		putBool(c, true);
		check("Bool into Cell<Dynamic>", Std.string(c.get()), "true");
		putString(c, "hi");
		check("String into Cell<Dynamic>", Std.string(c.get()), "hi");
		putInt(c, 7);
		check("Int into Cell<Dynamic>", Std.string(c.get()), "7");
		var half = 0.5;
		var made = new Cell<Dynamic>(half);
		check("constructor of Cell<Dynamic>", Std.string(made.get()), "0.5");
		Sys.println("CONFORMANCE_OK");
	}
}
