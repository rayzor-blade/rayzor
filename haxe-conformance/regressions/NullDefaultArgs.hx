// A null for a defaulted parameter takes the default, and a leading optional
// is skipped by type: a Float does not fill an Int slot.
class NullDefaultArgs {
	var inull:Null<Int> = null;
	var fnull:Null<Float> = null;
	var dnull:Dynamic = null;
	var ival:Null<Int> = 5;
	var fval:Null<Float> = 8.5;
	var dval:Dynamic = 6;

	public function new() {}

	function get(a = 2, b = 4.25) return "" + a + "/" + b;

	function getOpt(?a = 2, ?b = 4.25) return "" + a + "/" + b;

	static function sget(a = 2, b = 4.25) return "" + a + "/" + b;

	static function def(p) return def2(p);

	static inline function def2(?p = 10) return p;

	function run() {
		var got = [
			get(), get(5), get(5, 8.5), get(8.5), get(inull), get(inull, fnull), get(inull, inull), get(fnull),
			get(dnull, dnull), get(ival, fval), get(dval, dval), getOpt(8.5), getOpt(inull), getOpt(fnull),
			sget(8.5), sget(inull), sget(fnull), sget(null, 1.5), "" + def(null), "" + def(3)
		].join(" ");
		var want = "2/4.25 5/4.25 5/8.5 2/8.5 2/4.25 2/4.25 2/4.25 2/4.25 2/4.25 5/8.5 6/6 2/8.5 2/4.25 2/4.25 "
			+ "2/8.5 2/4.25 2/4.25 2/1.5 10 3";
		trace(got == want ? "CONFORMANCE_OK" : "FAIL " + got);
	}

	static function main() new NullDefaultArgs().run();
}
