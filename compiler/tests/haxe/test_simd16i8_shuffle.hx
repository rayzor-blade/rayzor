import rayzor.SIMD16i8;
import rayzor.SIMD4i32;
import rayzor.Bytes;
import rayzor.Ptr;

/**
 * Byte-lane shuffle (pshufb / tbl1 / i8x16.swizzle).
 *
 * Contract under test: `idx[i]` in 0..15 selects that lane; bit 7 set yields 0.
 * 16..127 is deliberately unspecified across targets and is NOT tested.
 */
class TestSimd16i8Shuffle {
	static var failures = 0;

	static function check(name:String, actual:Int, expected:Int):Void {
		if (actual != expected) {
			failures++;
			Sys.println("FAIL " + name + ": expected " + expected + " but got " + actual);
		}
	}

	static function main() {
		// Source vector: lane i = i * 3 (0,3,6,...,45) — distinct and < 128 so
		// sign-extension on read-back is not in play.
		var buf = Bytes.alloc(16);
		for (i in 0...16)
			buf.set(i, i * 3);
		var v = SIMD16i8.load(Ptr.fromRaw(buf.address()));

		// Reverse the lanes.
		var rev = SIMD16i8.make16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
		var r = SIMD16i8.shuffle(v, rev);
		for (i in 0...16)
			check("rev lane " + i, r.get(i), (15 - i) * 3);

		// Broadcast lane 5 to every lane.
		var bc = SIMD16i8.make16(5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5);
		var b = SIMD16i8.shuffle(v, bc);
		for (i in 0...16)
			check("bcast lane " + i, b.get(i), 15);

		// Bit 7 set must yield 0; other lanes still select.
		var zmask = SIMD16i8.make16(0x80, 1, 0x80, 3, 0x80, 5, 0x80, 7, 0x80, 9, 0x80, 11, 0x80, 13, 0x80, 15);
		var z = SIMD16i8.shuffle(v, zmask);
		for (i in 0...16)
			check("zero lane " + i, z.get(i), (i % 2 == 0) ? 0 : i * 3);

		// shuffleBytes: {j,0x80,0x80,0x80} x4 zero-extends byte j into all four
		// i32 lanes — the scale-broadcast pattern.
		var bcast7 = SIMD16i8.make16(7, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80);
		var w = SIMD4i32.shuffleBytes(v, bcast7);
		for (i in 0...4)
			check("i32 bcast lane " + i, w.get(i), 21);

		// SIMD4i32.load reads 4 contiguous i32.
		var ib = Bytes.alloc(16);
		for (i in 0...4)
			ib.setInt32(i * 4, 1000 + i);
		var lv = SIMD4i32.load(Ptr.fromRaw(ib.address()));
		for (i in 0...4)
			check("i32 load lane " + i, lv.get(i), 1000 + i);

		if (failures == 0)
			Sys.println("ALL PASS");
		else
			Sys.println("FAILURES: " + failures);
	}
}
