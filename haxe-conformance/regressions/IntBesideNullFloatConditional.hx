// `cond ? 0 : nullableFloat` is a Null<Float>: the Int is that Float, and the
// value reaches a Float slot or parameter as a number.
typedef IntBesideNullFloatMeta = {@:optional var focal:Float; final x1:Float;}

class IntBesideNullFloatConditional {
	static function show(f:Float):String
		return "" + f;

	static function main() {
		var absent:IntBesideNullFloatMeta = {x1: 1.0};
		var present:IntBesideNullFloatMeta = {x1: 1.0, focal: 0.25};
		var n:Null<Float> = null;
		var out:Array<Float> = [];
		out.push(n == null ? 0 : n);
		out.push(absent.focal == null ? 0 : absent.focal);
		out.push(present.focal == null ? 0 : present.focal);
		var v:Float = absent.focal == null ? 0 : absent.focal;
		out.push(v);
		var got = out.join(",") + " " + show(absent.focal == null ? 0 : absent.focal) + " " + show(present.focal != null ? present.focal : 1);
		if (got != "0,0,0.25,0 0 0.25")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
