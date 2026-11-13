// A scalar-replaced object carried into a nested loop leaves no pointer phi
// behind: an inline constructor chosen by a conditional, scaled per iteration.
class ScalarReplacedVec {
	public final x:Float;
	public final y:Float;

	public inline function new(x:Float, y:Float) {
		this.x = x;
		this.y = y;
	}

	public inline function scale(s:Float):ScalarReplacedVec
		return new ScalarReplacedVec(x * s, y * s);

	public inline function add(o:ScalarReplacedVec):ScalarReplacedVec
		return new ScalarReplacedVec(x + o.x, y + o.y);
}

class ScalarReplacedLoopCarried {
	static function main() {
		var out = [];
		for (f in [1, -1]) {
			var n = new ScalarReplacedVec(f, 0);
			var u = Math.abs(n.x) > 0.5 ? new ScalarReplacedVec(1, 2) : new ScalarReplacedVec(2, 1);
			for (c in [3, 4])
				out.push(n.add(u.scale(c)).x);
		}
		var got = out.join(",");
		if (got != "4,5,2,3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
