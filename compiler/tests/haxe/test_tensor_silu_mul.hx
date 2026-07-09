import rayzor.ds.DType;
import rayzor.ds.Tensor;

class Main {
    static function close(a:Float, b:Float):Bool {
        return Math.abs(a - b) < 0.00001;
    }

    static function silu(x:Float):Float {
        return x / (1.0 + Math.exp(-x));
    }

    static function main() {
        var a = Tensor.fromArray([-2.0, -0.5, 0.0, 1.0, 3.0], DType.F32);
        var b = Tensor.fromArray([2.0, -4.0, 10.0, 0.25, -0.5], DType.F32);
        var y = a.siluMul(b);
        var ok = true;
        for (i in 0...5) {
            ok = ok && close(y.getFlat(i), silu(a.getFlat(i)) * b.getFlat(i));
        }
        y.free();
        b.free();
        a.free();
        Sys.println(ok ? "PASS tensor-silu-mul" : "FAIL tensor-silu-mul");
    }
}
