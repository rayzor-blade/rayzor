// An array literal of Single, Int and Float values pushed into an
// Array<Array<Float>> field is an Array<Float>, not a Dynamic array of boxes.
class MixedNumberArrayLiteralShadow {
	public final layers:Array<Array<Float>> = [];

	public function new(x:Single, hex:Int, alpha:Single = 0.5, inset = false)
		layers.push([x, hex, alpha, inset ? 1 : 0, 2.25]);
}

class MixedNumberArrayLiteral {
	static function main() {
		var shadows = [for (i in 0...3) new MixedNumberArrayLiteralShadow(i, 0x6366f1)];
		var sum = 0.0;
		for (s in shadows)
			for (v in s.layers[0])
				sum += v;
		var l = shadows[2].layers[0];
		var got = l.length + "," + l[0] + "," + l[1] + "," + l[2] + "," + sum;
		if (got != "5,2,6514417,0.5," + (3 + 3 * 6514417 + 1.5 + 6.75))
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
