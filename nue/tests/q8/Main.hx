// Oracle test for the PRODUCTION pure-Haxe Q8_0 matmul (nue.Q4Matmul.matmul
// with a Q8_0 QTensor — the default path since e47f392f). Two assertions:
//
//  1. EXACT-INTEGER: activations are integer-valued with max|x| = 127, so the
//     kernel's symmetric int8 activation quantisation (scale = max|x|/127 = 1)
//     is lossless and the whole computation is exact in f32 for these
//     magnitudes. The result must match an f64 reference bit-for-bit at f32.
//
//  2. TOLERANCE vs the Rust reference kernel (matmulXTQThreaded) on the same
//     inputs. Deliberately NOT bit-exact: Haxe folds block products through a
//     SIMD4f accumulator, Rust in scalar f32 — legitimate rounding drift.
//
// Weight shape: rows=4, cols=160 -> 5 blocks/row: exercises BOTH the 4-wide
// SIMD4f block loop and the scalar tail block.
import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.ds.QTensor;
import nue.Q4Matmul;

class Main {
    // f32 rounding of an f64 value, via a 1-elem tensor round-trip.
    static function toF32(v:Float):Float {
        var t = Tensor.zeros([1], DType.F32);
        t.setFlat(0, v);
        return t.getFlat(0);
    }

    static function main() {
        var rows = 4;
        var cols = 160; // 5 blocks of 32
        var blocks = 5;

        // Build the Q8_0 buffer: per block, LE f16 scale then 32 int8.
        // Scales exactly representable in f16; weights a deterministic
        // full-i8-range pattern (Q8_0 uses all of -128..127 — this is why the
        // kernel must use SIMD4i32.dot, not the i7 variant).
        var scaleBits = [0x3800, 0x3C00, 0x4200, 0x3400, 0x4000]; // 0.5,1.0,3.0,0.25,2.0
        var scaleVals = [0.5, 1.0, 3.0, 0.25, 2.0];
        var buf = haxe.io.Bytes.alloc(rows * blocks * 34);
        var off = 0;
        for (r in 0...rows) {
            for (b in 0...blocks) {
                var bits = scaleBits[b];
                buf.set(off, bits & 0xFF);
                buf.set(off + 1, (bits >> 8) & 0xFF);
                for (j in 0...32) {
                    // pattern spans the full signed range incl. -128 and 127
                    var q = ((r * 37 + b * 11 + j * 7) % 256) - 128;
                    buf.set(off + 2 + j, q & 0xFF);
                }
                off += 34;
            }
        }
        var qw = QTensor.fromBytesQ8_0(buf, rows, cols);
        if (qw == null) { Sys.println("FAIL: fromBytesQ8_0 null"); Sys.exit(1); }

        // Integer-valued activation, max|x| = 127 -> lossless int8 quant.
        var x = Tensor.zeros([1, cols], DType.F32);
        for (i in 0...cols) {
            var v = ((i * 13) % 255) - 127; // -127..127, hits both extremes
            x.setFlat(i, v + 0.0);
        }
        x.setFlat(0, 127.0); // pin max|x| = 127 exactly

        // f64 reference from the same bytes (scale folded per block, exact).
        var expect = new Array<Float>();
        for (r in 0...rows) {
            var acc = 0.0;
            for (b in 0...blocks) {
                var dot = 0.0;
                var base = (r * blocks + b) * 34;
                for (j in 0...32) {
                    var q = buf.get(base + 2 + j);
                    if (q > 127) q -= 256;
                    dot += q * x.getFlat(b * 32 + j);
                }
                acc += scaleVals[b] * dot;
            }
            expect.push(acc);
        }

        // 1) Haxe kernel (default path).
        var yH = Q4Matmul.matmul(qw, x);
        var worstH = 0.0;
        for (r in 0...rows) {
            var d = yH.getFlat(r) - toF32(expect[r]);
            if (d < 0) d = -d;
            var m = expect[r] < 0 ? -expect[r] : expect[r];
            var rel = d / (m + 1.0);
            if (rel > worstH) worstH = rel;
        }
        Sys.println("haxe worst-rel=" + worstH);

        // 2) Rust reference on the same inputs.
        var xc = x.clone();
        var yR = qw.matmulXTQThreaded(xc, 0);
        var worstX = 0.0;
        for (r in 0...rows) {
            var d = yH.getFlat(r) - yR.getFlat(r);
            if (d < 0) d = -d;
            var m = yR.getFlat(r) < 0 ? -yR.getFlat(r) : yR.getFlat(r);
            var rel = d / (m + 1.0);
            if (rel > worstX) worstX = rel;
        }
        Sys.println("haxe-vs-rust worst-rel=" + worstX);

        // Exact path allows only f32 representation error; cross-kernel allows
        // fold-order drift.
        var ok = worstH < 0.000001 && worstX < 0.0001;
        Sys.println(ok ? "PASS" : "FAIL");
        Sys.exit(ok ? 0 : 1);
    }
}
