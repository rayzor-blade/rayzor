package b;

class Store {
	public static var last:Single = 0;

	public var a:Single;
	public var f:Float = 0;

	public function new(a:Single)
		this.a = a;

	public function setA(v:Single):Void
		a = v;

	public function setF(v:Single):Void
		f = v;

	public static function setLast(v:Single):Void
		last = v;
}

class Mixed {
	public var i:Int;
	public var s:Single;
	public var f:Float;

	public function new(i:Int, s:Single, f:Float) {
		this.i = i;
		this.s = s;
		this.f = f;
	}
}
