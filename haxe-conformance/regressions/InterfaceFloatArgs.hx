interface IBox<T> {
	function get():T;
	function set(val:T):Void;
}

class BoxS implements IBox<Single> {
	var v:Single;

	public function new(x:Single)
		v = x;

	public function get():Single
		return v;

	public function set(val:Single):Void
		v = val;
}

class BoxD implements IBox<Float> {
	var v:Float;

	public function new(x:Float)
		v = x;

	public function get():Float
		return v;

	public function set(val:Float):Void
		v = val;
}

@:forward(get, set)
abstract AbBox<T>(IBox<T>) from IBox<T> to IBox<T> {}

class InterfaceFloatArgs {
	static function check(label:String, got:Float, want:Float):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		var i = 3;
		var f = 4.5;
		var s:Single = 6;

		var w:IBox<Single> = new BoxS(50);
		w.set(70);
		check("Single, Int literal", w.get(), 70);
		w.set(70.5);
		check("Single, Float literal", w.get(), 70.5);
		w.set(i);
		check("Single, Int variable", w.get(), 3);
		w.set(f);
		check("Single, Float variable", w.get(), 4.5);
		w.set(s);
		check("Single, Single variable", w.get(), 6);
		var a:AbBox<Single> = new BoxS(1);
		a.set(8);
		check("Single, through a forwarding abstract", a.get(), 8);

		var d:IBox<Float> = new BoxD(50);
		d.set(70);
		check("Float, Int literal", d.get(), 70);
		d.set(i);
		check("Float, Int variable", d.get(), 3);
		d.set(s);
		check("Float, Single variable", d.get(), 6);
		var b:AbBox<Float> = new BoxD(1);
		b.set(9);
		check("Float, through a forwarding abstract", b.get(), 9);
		Sys.println("CONFORMANCE_OK");
	}
}
