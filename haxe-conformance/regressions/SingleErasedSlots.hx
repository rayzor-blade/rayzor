enum R<T> {
	Const(v:T);
	Bound(s:Array<T>);
}

@:forward
abstract Q<T>(R<T>) from R<T> to R<T> {
	@:from public static inline function fromConst<T>(val:T):Q<T>
		return Const(val);

	@:from public static inline function fromInt(val:Int):Q<Single>
		return Const((val:Single));
}

interface Counter {
	function get():Int;
	function set(v:Int):Void;
}

class Cell implements Counter {
	var a:Int;

	public function new(a:Int)
		this.a = a;

	public function get():Int
		return a;

	public function set(v:Int):Void
		a = v;
}

class Holder {
	public var f:Float = 0;

	public function new() {}

	public function store(v:Single):Void
		f = v;
}

class SingleErasedSlots {
	static function wrap<T>(v:T):R<T>
		return Const(v);

	static function id<T>(v:T):T
		return v;

	static function payload(r:R<Single>):Single {
		return switch (r) {
			case Const(v): v;
			default: -1;
		}
	}

	static function check(label:String, got:Float, want:Float):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		check("generic fn", payload(wrap((100:Single))), 100);
		var literal:R<Single> = Const(100.5);
		check("enum literal", payload(literal), 100.5);
		var s:Single = 4.5;
		check("identity", id(s), 4.5);

		var a:Array<Single> = [1.5, 2.5];
		a.push(3.5);
		var sum:Float = 0;
		for (x in a)
			sum += x;
		check("array iteration", sum, 7.5);
		check("array index", a[2], 3.5);
		check("array pop", a.pop(), 3.5);

		var m = new Map<String, Single>();
		m.set("k", 5.5);
		var v:Single = m.get("k");
		check("map get", v, 5.5);

		var h = new Holder();
		h.store(1.5);
		check("Single into Float field", h.f, 1.5);

		check("abstract from method", payload(Q.fromInt(100)), 100);
		var q:Q<Single> = 100;
		check("abstract implicit from", payload(q), 100);

		var c:Counter = cast new Cell(1);
		c.set(2);
		if (c.get() != 2)
			throw "cast of a new instance to an interface";
		var cell = new Cell(1);
		var d:Counter = cast cell;
		d.set(3);
		if (d.get() != 3 || cell.get() != 3)
			throw "cast of a local to an interface";

		trace("CONFORMANCE_OK");
	}
}
