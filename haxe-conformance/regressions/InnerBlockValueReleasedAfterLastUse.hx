// A value made inside a branch or loop body and read in later blocks is
// released after its last read, never before one.
class Cell {
	public var v:Int;

	public function new(v:Int)
		this.v = v;
}

class InnerBlockValueReleasedAfterLastUse {
	static function len(s:String):Int
		return s.length;

	static function branchy(i:Int):Int {
		var r = 0;
		if (i >= 0) {
			var k = "key" + i;
			if (i % 3 == 0)
				r = 100;
			r += k.length;
			if (i % 2 == 0)
				r += k.charCodeAt(0);
		}
		return r;
	}

	static function switched(i:Int):String {
		var out = "";
		if (i > 0) {
			var s = "v" + i;
			switch (i % 3) {
				case 0:
					out = "zero";
				case 1:
					out = "one";
				default:
					out = "many";
			}
			out += ":" + s;
		}
		return out;
	}

	static function objects(n:Int):Int {
		var total = 0;
		for (i in 0...n) {
			var c = new Cell(i);
			if (i % 2 == 0)
				total += 1;
			total += c.v;
			var a = [i, i + 1];
			if (i % 3 == 0)
				total += 10;
			total += a[1];
		}
		return total;
	}

	static function nested(n:Int):Int {
		var total = 0;
		for (i in 0...n) {
			var s = "n" + i;
			for (j in 0...2)
				total += j;
			total += len(s);
		}
		return total;
	}

	static function main() {
		final out:Array<String> = [];
		var b = 0;
		for (i in 0...20)
			b += branchy(i);
		out.push("" + b);
		out.push(switched(4) + "," + switched(5) + "," + switched(6) + "," + switched(0));
		out.push("" + objects(12));
		out.push("" + nested(15));
		final got = out.join(" | ");
		final want = "1860 | one:v4,many:v5,zero:v6, | 190 | 50";
		if (got != want)
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
