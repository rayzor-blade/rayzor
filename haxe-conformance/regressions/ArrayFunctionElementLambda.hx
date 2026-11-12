// An untyped function literal stored into an array of functions takes the
// element's parameter types.
import haxe.Int64;

class ArrayFunctionElementLambda {
	static final hooks:Array<(Int, Int64) -> Void> = [];

	static function main() {
		var seen:Array<String> = [];
		hooks.push((t, p) -> seen.push(t + ":" + Int64.toStr(p)));
		for (hook in hooks)
			hook(2, Int64.make(1, 8));
		var scaled:Array<(Int, Float) -> Float> = [];
		scaled.push((n, f) -> n * f);
		scaled.unshift((n, f) -> n + f);
		var lengths:Array<(String) -> Int> = [];
		lengths.insert(0, s -> s.length);
		var got = seen.join(",") + "|" + scaled[1](2, 2.5) + "|" + scaled[0](2, 2.5) + "|" + lengths[0]("abcd");
		if (got != "2:4294967304|5|4.5|4")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
