import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        var t = Tensor.uninit([2, 3], DType.F32);
        for (i in 0...6) {
            t.setFlat(i, i + 1);
        }
        var s = t.sum();
        t.free();

        if (Math.abs(s - 21.0) < 0.000001) {
            Sys.println("PASS tensor-uninit-full-overwrite sum=" + s);
        } else {
            Sys.println("FAIL tensor-uninit-full-overwrite sum=" + s);
        }
    }
}
