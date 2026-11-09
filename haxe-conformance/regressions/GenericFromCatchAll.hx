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

abstract ConstFirst<T>(RR<T>) from RR<T> to RR<T> {
	@:from static inline function fromConst<T>(val:T):ConstFirst<T>
		return K(val);

	@:from static inline function fromState<T>(state:Sg<T>):ConstFirst<T>
		return B(state);
}

abstract StateFirst<T>(RR<T>) from RR<T> to RR<T> {
	@:from static inline function fromState<T>(state:Sg<T>):StateFirst<T>
		return B(state);

	@:from static inline function fromConst<T>(val:T):StateFirst<T>
		return K(val);
}

typedef Attrs = {
	?width:ConstFirst<Int>
}

class GenericFromCatchAll {
	static function tagC(r:ConstFirst<Int>):String {
		return switch (r) {
			case B(s): "B" + s.ptr;
			case K(v): "K" + v;
		}
	}

	static function tagS(r:StateFirst<Int>):String {
		return switch (r) {
			case B(s): "B" + s.ptr;
			case K(v): "K" + v;
		}
	}

	static function check(label:String, got:String, want:String):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		var s:Sg<Int> = new SgI(2);
		check("catch-all declared first", tagC(s), "B77");
		check("catch-all declared last", tagS(s), "B77");
		check("constant", tagC(5), "K5");
		var a:Attrs = {width: s};
		check("typedef field", tagC(a.width), "B77");
		var r:ConstFirst<Int> = s;
		check("var initializer", tagC(r), "B77");
		Sys.println("CONFORMANCE_OK");
	}
}
