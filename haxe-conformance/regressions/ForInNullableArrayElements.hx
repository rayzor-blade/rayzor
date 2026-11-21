// `for (v in p)` over a `Null<Array<T>>` reads each element as a T, in a loop
// and in a comprehension.
class ForInNullableArrayElements {
	static function floats():Null<Array<Float>>
		return [0.25, 0.1, 0.25, 1];

	static function show(x:Float):String
		return "<" + Std.string(x) + ">";

	static function main() {
		var p = floats();
		var out = [];
		for (v in p)
			out.push(show(v));
		var sum = 0.0;
		for (v in p)
			sum += v;
		out.push(Std.string(sum));
		out.push([for (v in p) v * 2].join(" "));
		var ints:Null<Array<Int>> = [1, 2, 3];
		var total = 0;
		for (v in ints)
			total += v;
		out.push(Std.string(total));
		var words:Null<Array<String>> = ["a", "bc"];
		for (w in words)
			out.push(w + w.length);
		var flags:Null<Array<Bool>> = [true, false];
		for (f in flags)
			out.push(f ? "t" : "f");
		var got = out.join(",");
		if (got != "<0.25>,<0.1>,<0.25>,<1>,1.6,0.5 0.2 0.5 2,6,a1,bc2,t,f")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
