import haxe.io.Bytes;

// Bytes made and used inside one call are released when it ends; a view
// that outlives its owner, and bytes kept in a field, still read correctly.
class Holder {
	public var kept:Bytes;

	public function new() {}
}

class BytesLocalsReleased {
	static function local(i:Int):Int {
		var b = Bytes.alloc(64);
		b.set(1, i & 255);
		b.setInt32(4, i * 3);
		return b.get(1) + b.getInt32(4);
	}

	static function tail(i:Int):Bytes {
		var b = Bytes.alloc(8);
		b.set(7, i);
		return b.sub(4, 4);
	}

	static function copied(i:Int):Int {
		var src = Bytes.ofString("ab" + i);
		var dst = Bytes.alloc(src.length);
		dst.blit(0, src, 0, src.length);
		var same = dst.compare(src) == 0;
		return (same ? 100 : 0) + dst.toString().length;
	}

	static function branchy(i:Int):Int {
		var r = 0;
		if (i > 1) {
			var b = Bytes.alloc(4);
			b.fill(0, 4, i);
			if (i % 2 == 0)
				r += 1000;
			r += b.get(3);
		}
		return r;
	}

	static function keep(h:Holder, i:Int):Void {
		var b = Bytes.alloc(2);
		b.set(0, i);
		h.kept = b;
	}

	static function main() {
		var a = 0, t = 0, c = 0, br = 0;
		var h = new Holder();
		for (i in 0...6) {
			a += local(i);
			t += tail(i).get(3);
			c += copied(i);
			br += branchy(i);
			keep(h, i + 40);
		}
		final got = a + "," + t + "," + c + "," + br + "," + h.kept.get(0);
		final want = "60,15,618,2014,45";
		if (got != want)
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
