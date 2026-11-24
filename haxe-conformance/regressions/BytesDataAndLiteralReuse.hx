import haxe.io.Bytes;

// A `getData()` copy read in place is released after its last read, and a
// string literal evaluated many times reads the same wherever it went.
class Holder {
	public var label:String = "";

	public function new() {}
}

class BytesDataAndLiteralReuse {
	static function crcOf(i:Int):Int {
		var b = Bytes.alloc(16);
		for (k in 0...16)
			b.set(k, (i + k) & 255);
		return haxe.crypto.Crc32.make(b);
	}

	static function sumData(i:Int):Int {
		var b = Bytes.alloc(8);
		b.set(3, i);
		var d = b.getData();
		var t = 0;
		for (k in 0...8)
			t += Bytes.fastGet(d, k);
		return t;
	}

	static function tag(i:Int):String
		return i % 2 == 0 ? "even" : "odd";

	static function main() {
		var crc = 0;
		var data = 0;
		var counts = new Map<String, Int>();
		var h = new Holder();
		var same = 0;
		for (i in 0...10) {
			crc ^= crcOf(i);
			data += sumData(i);
			var t = tag(i);
			counts.set(t, (counts.exists(t) ? counts.get(t) : 0) + 1);
			h.label = "last";
			if ("last" == h.label && t.length > 2)
				same++;
		}
		final got = crc + "," + data + "," + counts.get("even") + "," + counts.get("odd") + "," + h.label + "," + same;
		final want = "-409204561,45,5,5,last,10";
		if (got != want)
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
