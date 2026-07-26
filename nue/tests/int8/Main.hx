// Oracle for the PRODUCTION pure-Haxe INT8 (per-row) matmul — nue.Q4Matmul.matmul
// with an INT8 QTensor, the band kernel that replaced the matmulXTQThreaded FFI
// bail. Two assertions on the SAME quantised tensor:
//
//  1. vs DEQUANT reference: activations are integer-valued with max|x| = 127, so
//     symmetric int8 activation quant is lossless (scale = max|x|/127 = 1). The
//     kernel then computes exactly SUM_k dequant[n,k] * x[k], so its output must
//     match an f64 dot of the tensor's OWN dequantised weights against x, at f32.
//
//  2. TOLERANCE vs the Rust reference kernel (matmulXTQThreaded) — proves the
//     Haxe kernel reproduces the kernel it replaced. Not bit-exact: Haxe folds
//     through SDOT super-chunks, Rust in its own order.
//
// cols = 216 = 6*32 + 16 + 8: exercises the 32-element super-chunk loop, the
// trailing 16-lane chunk, AND the scalar tail (single signed-byte loads).
import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.ds.QTensor;
import rayzor.ds.QScheme;
import nue.Q4Matmul;

class Main {
    static function main() {
        var rows = 8;
        var cols = 216;

        // Weights: arbitrary but deterministic f32 (fromFloat32 picks a per-row
        // symmetric scale). A wide spread per row so the int8 range is used.
        var w = Tensor.zeros([rows, cols], DType.F32);
        for (r in 0...rows) {
            for (c in 0...cols) {
                var v = (((r * 131 + c * 17) % 509) - 254) * 0.37;
                w.setFlat(r * cols + c, v);
            }
        }
        var qw = QTensor.fromFloat32(w, QScheme.INT8);
        if (qw == null) { Sys.println("FAIL: fromFloat32 null"); Sys.exit(1); }
        if (qw.scheme() != QScheme.INT8) { Sys.println("FAIL: not INT8"); Sys.exit(1); }

        // Integer activation, max|x| = 127 -> lossless int8 quant.
        var x = Tensor.zeros([1, cols], DType.F32);
        for (i in 0...cols) {
            var v = ((i * 13) % 255) - 127; // -127..127
            x.setFlat(i, v + 0.0);
        }
        x.setFlat(0, 127.0); // pin max|x| = 127 exactly

        // Dequant reference: the tensor's OWN int8 weights * scale, dotted with x
        // in f64. With x lossless this is exactly what the kernel must produce.
        var dq = qw.dequant();
        var expect = new Array<Float>();
        for (r in 0...rows) {
            var acc = 0.0;
            for (c in 0...cols) acc += dq.getFlat(r * cols + c) * x.getFlat(c);
            expect.push(acc);
        }

        // 1) Haxe kernel (default INT8 path).
        var yH = Q4Matmul.matmul(qw, x);
        var worstH = 0.0;
        for (r in 0...rows) {
            var d = yH.getFlat(r) - toF32(expect[r]);
            if (d < 0) d = -d;
            var m = expect[r] < 0 ? -expect[r] : expect[r];
            var rel = d / (m + 1.0);
            if (rel > worstH) worstH = rel;
        }
        Sys.println("haxe-vs-dequant worst-rel=" + worstH);

        // 2) Rust reference kernel on the same inputs.
        var yR = qw.matmulXTQThreaded(x.clone(), 0);
        var worstX = 0.0;
        for (r in 0...rows) {
            var d = yH.getFlat(r) - yR.getFlat(r);
            if (d < 0) d = -d;
            var m = yR.getFlat(r) < 0 ? -yR.getFlat(r) : yR.getFlat(r);
            var rel = d / (m + 1.0);
            if (rel > worstX) worstX = rel;
        }
        Sys.println("haxe-vs-rust worst-rel=" + worstX);

        // Dequant path allows only f32 representation error; cross-kernel allows
        // fold-order drift.
        var ok = worstH < 0.00001 && worstX < 0.0001;
        Sys.println(ok ? "PASS" : "FAIL");
        Sys.exit(ok ? 0 : 1);
    }

    // f32 rounding of an f64 value, via a 1-elem tensor round-trip.
    static function toF32(v:Float):Float {
        var t = Tensor.zeros([1], DType.F32);
        t.setFlat(0, v);
        return t.getFlat(0);
    }
}
