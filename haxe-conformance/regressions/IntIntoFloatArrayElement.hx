// An Int stored into a Float or Single array element is that number, not its
// bits: a literal, a conditional of literals, or an Int expression.
class IntIntoFloatArrayElement {
	static final ny:Array<Float> = [];

	static function main() {
		for (i in 0...3)
			ny.push(0);
		ny[0] = -1;
		ny[1] = 1;
		var below = Sys.args().length == 0 ? 1.0 : -1.0;
		ny[2] = below > 0 ? -1 : 1;
		var local:Array<Float> = [0.5, 0.5];
		local[0] += 1;
		var i = 4;
		local[1] = i * 2;
		var s:Array<Single> = [0, 0];
		s[0] = 3;
		s[1] = below > 0 ? -2 : 7;
		var got = ny.join(",") + " " + (ny[0] * 0.5) + " " + local.join(",") + " " + s.join(",");
		if (got != "-1,1,-1 -0.5 1.5,8 3,-2")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
