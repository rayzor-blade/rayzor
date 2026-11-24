// Values handed only to runtime reads (a map lookup, parseInt, an element
// read, a join) are released after the read, and the read sees them intact.
class ReadOnlyRuntimeCallArguments {
	static final names = ["a" => 1, "b2" => 2, "c33" => 3];

	static function lookup(i:Int):Int {
		var key = "b" + i;
		var hit = names.get(key);
		var there = names.exists(key);
		return (hit == null ? 0 : hit) + (there ? 10 : 0);
	}

	static function parsed(i:Int):Int {
		var text = "" + (i * 7);
		var n = Std.parseInt(text);
		var f = Std.parseFloat(text + ".5");
		return n + Std.int(f * 2);
	}

	static function elements(i:Int):Int {
		var xs = [i, i + 1, i + 2];
		xs[1] = xs[0] + xs[2];
		xs.push(i * 3);
		return xs[1] + xs[3] + xs.join("-").length;
	}

	static function removed(i:Int):Int {
		var m = new Map<String, Int>();
		m.set("k" + i, i);
		var key = "k" + i;
		var had = m.remove(key);
		return had ? 1 : 0;
	}

	static function main() {
		var a = 0, b = 0, c = 0, d = 0;
		for (i in 0...5) {
			a += lookup(i);
			b += parsed(i);
			c += elements(i);
			d += removed(i);
		}
		final got = a + "," + b + "," + c + "," + d;
		final want = "12,215,97,5";
		if (got != want)
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
