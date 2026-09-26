// Abstract methods whose bodies the inliner cannot substitute through (a
// block, a closure over `this`, a call taking `this`) are called instead of
// inlined, so `this` is bound and the statement using them runs.
private abstract Num(Int) {
	public function new(v:Int) this = v;

	public function block():Int return {
		var y = this;
		y + 1;
	};

	public function thunk():Void->Int return () -> this * 2;

	public function viaStatic():Int {
		return twice(this);
	}

	static function twice(x:Int) return x * 2;
}

class AbstractMethodBodies {
	static function main() {
		var n = new Num(5);
		var got = n.block() + " " + n.thunk()() + " " + n.viaStatic();
		if (got == "6 10 10")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
