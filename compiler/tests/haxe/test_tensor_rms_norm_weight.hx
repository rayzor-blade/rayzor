import rayzor.ds.DType;
import rayzor.ds.Tensor;

class Main {
    static function close(a:Float, b:Float):Bool {
        return Math.abs(a - b) < 0.00001;
    }

    static function main() {
        var base = Tensor.fromArray([1.0, -2.0, 3.0, -4.0], DType.F32);
        var x = base.reshape([1, 4]);
        var weight = Tensor.fromArray([0.5, 0.25, 1.0, 2.0], DType.F32);
        var y = x.rmsNormWeight(weight, 0.000001);

        var inv = 1.0 / Math.sqrt((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + 0.000001);
        var expected = [
            1.0 * 0.5 * inv,
            -2.0 * 0.25 * inv,
            3.0 * 1.0 * inv,
            -4.0 * 2.0 * inv
        ];

        var ok = true;
        for (i in 0...4) {
            ok = ok && close(y.getFlat(i), expected[i]);
        }

        y.free();
        weight.free();
        x.free();
        base.free();

        if (ok) {
            Sys.println("PASS tensor-rms-norm-weight");
        } else {
            Sys.println("FAIL tensor-rms-norm-weight");
        }
    }
}
