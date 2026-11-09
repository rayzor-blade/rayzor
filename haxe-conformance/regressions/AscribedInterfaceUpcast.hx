interface ISg<T> {
	var ptr(default, null):Int;
	function get():T;
}

class SgI implements ISg<Int> {
	public var ptr(default, null):Int;

	var v:Int;

	public function new(v:Int) {
		this.v = v;
		ptr = 77;
	}

	public function get():Int
		return v;
}

@:forward(ptr, get)
abstract Sg<T>(ISg<T>) from ISg<T> to ISg<T> {}

enum RR<T> {
	K(v:T);
	B(s:Sg<T>);
}

@:forward
abstract IRR<T>(RR<T>) from RR<T> to RR<T> {
	@:from public static inline function fromState<T>(s:Sg<T>):IRR<T>
		return B(s);
}

class AscribedInterfaceUpcast {
	static function ptrOf(r:IRR<Int>):Int {
		return switch (r) {
			case B(s): s.ptr;
			case _: -1;
		}
	}

	static function check(label:String, got:Int, want:Int):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		var t = (new SgI(2) : Sg<Int>);
		check("ascribed abstract property", t.ptr, 77);
		check("ascribed abstract method", t.get(), 2);
		check("ascribed interface", (new SgI(3) : ISg<Int>).get(), 3);
		check("ascribed into @:from", ptrOf(IRR.fromState((new SgI(2) : Sg<Int>))), 77);
		check("ascribed local into @:from", ptrOf(IRR.fromState(t)), 77);
		Sys.println("CONFORMANCE_OK");
	}
}
