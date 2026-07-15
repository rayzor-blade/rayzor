import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.Bytes;
import nue.Int8Matmul;

/**
 * Correctness harness for the INT8 VNNI GEMM. Compares int8 GEMM vs exact-F32
 * matmulT (run with RZT_AMX_PREFILL=0 so the reference is exact F32).
 */
class Main {
    static function main() {
        checkAll();
    }

    static function checkAll():Void {
        check(32, 384, 384);
        check(40, 384, 1536);
        check(40, 1536, 384);
    }

    static function check(m:Int, k:Int, n:Int):Void {
        var wData = [for (i in 0...n * k) Math.sin(i * 0.017 + 0.3) * 0.5];
        var xData = [for (i in 0...m * k) Math.cos(i * 0.013 + 0.1) * 1.3];
        var w = Tensor.fromArray(wData, F32).reshape([n, k]);
        var x = Tensor.fromArray(xData, F32).reshape([m, k]);
        var yRef = x.matmulT(w);

        var qw = Bytes.alloc(n * k);
        var wScale = Bytes.alloc(n * 4);
        var wSum = Bytes.alloc(n * 4);
        Int8Matmul.quantizeWeight(w, n, k, qw.address(), wScale.address(), wSum.address());
        var y8 = Int8Matmul.matmul(x, m, k, n, qw.address(), wScale.address(), wSum.address());

        var dot = 0.0;
        var na = 0.0;
        var nb = 0.0;
        for (idx in 0...m * n) {
            var a = yRef.getFlat(idx);
            var b = y8.getFlat(idx);
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        var cos = dot / (Math.sqrt(na) * Math.sqrt(nb));
        Sys.println("[m=" + m + " k=" + k + " n=" + n + "] cosine=" + cos
            + "  " + (cos > 0.99 ? "PASS" : "FAIL"));

        w.free();
        x.free();
        yRef.free();
        y8.free();
        qw.free();
        wScale.free();
        wSum.free();
    }
}
