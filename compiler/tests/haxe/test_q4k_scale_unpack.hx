import rayzor.SIMD16i8;
import rayzor.SIMD4i32;
import rayzor.Bytes;
import rayzor.Ptr;

/**
 * The Q4_K 6-bit scale/min unpack, done in vector lanes, must be bit-identical
 * to the scalar shift/mask form the kernel uses today.
 *
 * Layout: header bytes 4..15 are ggml's q[0..11].
 *   sc[j]   = q[j] & 63                                   for j in 0..3
 *   mn[j]   = q[j+4] & 63                                 for j in 0..3
 *   sc[j+4] = (q[j+8] & 0x0F) | ((q[j]   >> 6) << 4)      for j in 0..3
 *   mn[j+4] = (q[j+8] >> 4)   | ((q[j+4] >> 6) << 4)      for j in 0..3
 *
 * The vector form lands sc0..7 in lanes 0..7 and mn0..7 in lanes 8..15, using
 * that (x >> 6) << 4 == (x >> 2) & 0x30.
 */
class TestQ4kScaleUnpack {
	static var failures = 0;

	static function main() {
		var hdr = Bytes.alloc(16);
		// Deterministic pseudo-random headers; every q byte spans 0..255 so the
		// 2-bit high fields and the nibble fields are all exercised.
		var seed = 12345;
		for (trial in 0...64) {
			for (i in 0...16) {
				seed = (seed * 1103515245 + 12345) & 0x3FFFFFFF;
				hdr.set(i, (seed >> 7) & 0xFF);
			}

			// --- scalar reference, exactly as q4DotMA4 derives it ---
			var u0 = hdr.getInt32(4);
			var u1 = hdr.getInt32(8);
			var u2 = hdr.getInt32(12);
			var sc = [
				u0 & 63, (u0 >>> 8) & 63, (u0 >>> 16) & 63, (u0 >>> 24) & 63,
				(u2 & 0x0F) | (((u0 >>> 6) & 3) << 4),
				((u2 >>> 8) & 0x0F) | (((u0 >>> 14) & 3) << 4),
				((u2 >>> 16) & 0x0F) | (((u0 >>> 22) & 3) << 4),
				((u2 >>> 24) & 0x0F) | (((u0 >>> 30) & 3) << 4)
			];
			var mn = [
				u1 & 63, (u1 >>> 8) & 63, (u1 >>> 16) & 63, (u1 >>> 24) & 63,
				((u2 >>> 4) & 0x0F) | (((u1 >>> 6) & 3) << 4),
				((u2 >>> 12) & 0x0F) | (((u1 >>> 14) & 3) << 4),
				((u2 >>> 20) & 0x0F) | (((u1 >>> 22) & 3) << 4),
				((u2 >>> 28) & 0x0F) | (((u1 >>> 30) & 3) << 4)
			];

			// --- vector form ---
			var H = SIMD16i8.load(Ptr.fromRaw(hdr.address()));
			var A = SIMD16i8.shuffle(H, SIMD16i8.make16(4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15));
			var B = SIMD16i8.shuffle(H, SIMD16i8.make16(0x80, 0x80, 0x80, 0x80, 4, 5, 6, 7, 0x80, 0x80, 0x80, 0x80, 8, 9, 10, 11));
			var lo = SIMD16i8.or(SIMD16i8.and(A, SIMD16i8.make16(63, 63, 63, 63, 15, 15, 15, 15, 63, 63, 63, 63, 0, 0, 0, 0)), SIMD16i8.and(SIMD16i8.ushr(A, 4), SIMD16i8.make16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 15, 15)));
			var hi = SIMD16i8.and(SIMD16i8.ushr(B, 2), SIMD16i8.splat(0x30));
			var S = SIMD16i8.or(lo, hi);

			for (j in 0...8) {
				var gotSc = S.get(j) & 0xFF;
				if (gotSc != sc[j]) {
					failures++;
					if (failures < 8)
						Sys.println("FAIL trial " + trial + " sc[" + j + "]: expected " + sc[j] + " got " + gotSc);
				}
				var gotMn = S.get(8 + j) & 0xFF;
				if (gotMn != mn[j]) {
					failures++;
					if (failures < 8)
						Sys.println("FAIL trial " + trial + " mn[" + j + "]: expected " + mn[j] + " got " + gotMn);
				}
			}

			// The scale broadcast used by the dot: byte j zero-extended into
			// all four i32 lanes.
			if (trial == 0) {
				var bc3 = SIMD16i8.make16(3, 0x80, 0x80, 0x80, 3, 0x80, 0x80, 0x80, 3, 0x80, 0x80, 0x80, 3, 0x80, 0x80, 0x80);
				var v = SIMD4i32.shuffleBytes(S, bc3);
				for (l in 0...4)
					if (v.get(l) != sc[3]) {
						failures++;
						Sys.println("FAIL broadcast lane " + l + ": expected " + sc[3] + " got " + v.get(l));
					}
			}
		}

		if (failures == 0)
			Sys.println("ALL PASS");
		else
			Sys.println("FAILURES: " + failures);
	}
}
