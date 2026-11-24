// A map made and used inside one call is released when it ends; the values
// it held, a map returned or kept in a field, and a copy all stay intact.
class Point {
	public var x:Int;

	public function new(x:Int)
		this.x = x;
}

class Keeper {
	public var names:Map<String, Int>;

	public function new() {
		names = new Map();
	}
}

class LocalMapsReleased {
	static function fill(m:Map<String, Int>, i:Int):Void {
		m.set("a" + i, i);
		m.set("b", i * 2);
	}

	static function counted(i:Int):Int {
		var m = new Map<String, Int>();
		fill(m, i);
		var total = 0;
		for (k in m.keys())
			total += m.get(k);
		return total + (m.exists("b") ? 100 : 0);
	}

	static function ints(i:Int):Int {
		var m = new Map<Int, String>();
		m.set(i, "v" + i);
		m.set(i + 1, "w");
		var copy = m.copy();
		m.remove(i);
		return (m.exists(i) ? 1 : 0) + copy.get(i).length;
	}

	static function picked(i:Int):Point {
		var m = new Map<Point, Int>();
		var p = new Point(i);
		var q = new Point(i * 10);
		m.set(p, 1);
		m.set(q, 2);
		return m.get(q) == 2 ? q : p;
	}

	static function made(i:Int):Map<String, Int> {
		var m = new Map<String, Int>();
		m.set("k", i);
		return m;
	}

	static function main() {
		var a = 0, b = 0, c = 0, d = 0;
		var keep = new Keeper();
		for (i in 0...5) {
			a += counted(i);
			b += ints(i);
			c += picked(i).x;
			d += made(i).get("k");
			keep.names.set("n" + i, i);
		}
		final got = a + "," + b + "," + c + "," + d + "," + keep.names.get("n4");
		final want = "530,10,100,10,4";
		if (got != want)
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
