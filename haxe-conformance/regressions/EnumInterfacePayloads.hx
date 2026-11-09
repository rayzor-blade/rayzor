interface IPlain {
	function id():Int;
}

class PlainImpl implements IPlain {
	public function new() {}

	public function id():Int
		return 77;
}

enum RP {
	BP(s:IPlain);
}

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

enum RS {
	BS(s:Sg<Int>);
}

class EnumInterfacePayloads {
	static function idRP(r:RP):Int
		return switch r {
			case BP(s): s.id();
		};

	static function ptrRS(r:RS):Int
		return switch r {
			case BS(s): s.ptr;
		};

	static function getRS(r:RS):Int
		return switch r {
			case BS(s): s.get();
		};

	static function check(label:String, got:Int, want:Int):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		check("class into an interface payload", idRP(BP(new PlainImpl())), 77);
		var i:IPlain = new PlainImpl();
		check("interface value into a payload", idRP(BP(i)), 77);
		var r = BP(new PlainImpl());
		check("local enum of a class payload", switch r {
			case BP(s): s.id();
		}, 77);
		var s:Sg<Int> = new SgI(2);
		check("abstract payload property", ptrRS(BS(s)), 77);
		check("abstract payload from a class", ptrRS(BS(new SgI(2))), 77);
		check("abstract payload method", getRS(BS(s)), 2);
		Sys.println("CONFORMANCE_OK");
	}
}
