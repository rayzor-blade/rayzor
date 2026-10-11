// A plain Int parameter's default is applied by the callee: an omitted or
// null argument through a function value, a base-class or interface
// reference, `super` or Reflect gets the implementation's own default.
interface IDef {
	function get(x:Int = 5):Int;
}

class BaseDef {
	public function get(x = 3) return x;
}

class ExtDef extends BaseDef implements IDef {
	public function new() {}

	override function get(x = 7) return x;
}

class Ext2Def extends ExtDef {
	override function get(x = 9) return super.get() + x;
}

class CalleeIntDefaults {
	public function new() {}

	function opt(x = 10):Null<Int> return x + 1;

	static function sopt(a:Int, x = 20):Int return a + x;

	function neg(x = -3) return x;

	function run() {
		var o:?Int->Null<Int> = opt;
		var s:Int->?Int->Int = sopt;
		var e = new ExtDef();
		var b:BaseDef = e;
		var i:IDef = e;
		var b2:BaseDef = new Ext2Def();
		var got = [
			opt(), opt(null), opt(5), o(), o(3), o(null), s(1), s(1, 2), sopt(1), e.get(), b.get(), b.get(9),
			i.get(), i.get(-4), b2.get(), neg(), neg(0), Reflect.callMethod(this, opt, []), Reflect.callMethod(this, opt, [0]),
			Reflect.callMethod(null, sopt, [1])
		].join(" ");
		var want = "11 11 6 11 4 11 21 3 21 7 7 9 7 -4 16 -3 0 11 1 21";
		trace(got == want ? "CONFORMANCE_OK" : "FAIL " + got);
	}

	static function main() new CalleeIntDefaults().run();
}
