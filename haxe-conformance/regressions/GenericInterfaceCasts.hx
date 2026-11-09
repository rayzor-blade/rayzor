interface IG<T> {
	function get():T;
	function set(v:T):Void;
}

class CG implements IG<Int> {
	var a:Int;

	public function new(a:Int)
		this.a = a;

	public function get():Int
		return a;

	public function set(v:Int):Void
		a = v;
}

class Base<T> implements IG<T> {
	var v:T;

	public function new(v:T)
		this.v = v;

	public function get():T
		return v;

	public function set(v:T):Void
		this.v = v;
}

class Sub extends Base<String> {}

@:forward(get, set)
abstract AG<T>(IG<T>) from IG<T> to IG<T> {
	public static function make(v:Int):AG<Int>
		return (cast new CG(cast v) : AG<Int>);
}

class GenericInterfaceCasts {
	static function check(value:Bool, label:String):Void {
		if (!value)
			throw label;
	}

	static function main():Void {
		var x:IG<Int> = cast new CG(1);
		x.set(2);
		check(x.get() == 2, "cast new to a generic interface");

		var c = new CG(3);
		var y:IG<Int> = cast c;
		y.set(4);
		check(y.get() == 4 && c.get() == 4, "cast a variable to a generic interface");

		var z = (cast new CG(5) : IG<Int>);
		z.set(6);
		check(z.get() == 6, "typed cast to a generic interface");

		var s = (cast new Sub("a") : IG<String>);
		s.set("b");
		check(s.get() == "b", "cast through an inherited generic interface");

		var a = AG.make(7);
		a.set(8);
		check(a.get() == 8, "cast to an abstract over a generic interface");

		trace("CONFORMANCE_OK");
	}
}
