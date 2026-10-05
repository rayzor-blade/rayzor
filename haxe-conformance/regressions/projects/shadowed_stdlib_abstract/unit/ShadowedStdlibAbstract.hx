package unit;

// A private type named like a stdlib type keeps its own methods, and the
// stdlib type stays usable by its full name in the same module.
private abstract Int64(Int) {
	inline function new(i:Int) {
		this = i;
	}

	public static function make(a:Int, b:Int):Int64 {
		return new Int64(a + b);
	}

	@:op(A * B) public function mul(o:Int64):Int64 {
		return new Int64(this * o.toInt());
	}

	public function toInt():Int {
		return this;
	}
}

class ShadowedStdlibAbstract {
	static function main() {
		var c = Int64.make(1, 2) * Int64.make(3, 4);
		var h = haxe.Int64.make(1, 2);
		var ok = c.toInt() == 21 && haxe.Int64.toStr(h) == "4294967298";
		trace(ok ? "CONFORMANCE_OK" : "FAIL " + c.toInt() + " " + haxe.Int64.toStr(h));
	}
}
