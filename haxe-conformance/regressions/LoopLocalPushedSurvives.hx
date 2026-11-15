class P {
	public var n:Int;

	public function new(n:Int) {
		this.n = n;
	}
}

class Holder {
	public final kept:Array<P> = [];

	public function new() {}

	public function keep(p:P):Void
		kept.push(p);
}

// A local made in a loop body and handed on (pushed, or passed to a method
// that keeps it) outlives the body: each iteration's object is its own.
class LoopLocalPushedSurvives {
	static function main() {
		var ps:Array<P> = [];
		for (k in 0...3) {
			var line = new P(k + 1);
			ps.push(line);
		}
		var h = new Holder();
		var i = 0;
		while (i < 2) {
			var p = new P(10 * (i + 1));
			h.keep(p);
			i++;
		}
		var later = new P(99);
		var got = ps.map(p -> p.n).join(",") + " " + (ps[0] == ps[1]) + " " + h.kept.map(p -> p.n).join(",") + " " + later.n;
		if (got != "1,2,3 false 10,20 99")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
