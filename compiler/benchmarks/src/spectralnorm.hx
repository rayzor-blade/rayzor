// Spectral-norm benchmark — the Computer Language Benchmarks Game algorithm,
// single-threaded. Prints the norm scaled to an integer (x 1e9), since float
// formatting differs between targets.
//
// Tests: Float arithmetic, division, Float array traversal

package benchmarks;

class SpectralNorm {
    static inline var N = 2000;

    static inline function a(i:Int, j:Int):Float {
        var ij = i + j;
        return 1.0 / (ij * (ij + 1) * 0.5 + i + 1);
    }

    static function multiplyAv(n:Int, v:Array<Float>, av:Array<Float>) {
        for (i in 0...n) {
            var sum = 0.0;
            for (j in 0...n)
                sum += a(i, j) * v[j];
            av[i] = sum;
        }
    }

    static function multiplyAtv(n:Int, v:Array<Float>, atv:Array<Float>) {
        for (i in 0...n) {
            var sum = 0.0;
            for (j in 0...n)
                sum += a(j, i) * v[j];
            atv[i] = sum;
        }
    }

    static function multiplyAtAv(n:Int, v:Array<Float>, out:Array<Float>, tmp:Array<Float>) {
        multiplyAv(n, v, tmp);
        multiplyAtv(n, tmp, out);
    }

    public static function main() {
        var n = N;
        var u = [for (i in 0...n) 1.0];
        var v = [for (i in 0...n) 0.0];
        var tmp = [for (i in 0...n) 0.0];
        for (i in 0...10) {
            multiplyAtAv(n, u, v, tmp);
            multiplyAtAv(n, v, u, tmp);
        }
        var vBv = 0.0;
        var vv = 0.0;
        for (i in 0...n) {
            vBv += u[i] * v[i];
            vv += v[i] * v[i];
        }
        Sys.println("spectralnorm x1e9 = " + Std.int(Math.sqrt(vBv / vv) * 1e9));
    }
}
