// Fannkuch-redux benchmark — the Computer Language Benchmarks Game algorithm,
// single-threaded. n = 10: checksum 73196, max flips 38.
//
// Tests: Int array indexing, tight nested loops, permutation generation

package benchmarks;

class FannkuchRedux {
    static inline var N = 10;

    public static function fannkuch(n:Int):Array<Int> {
        var perm = [for (i in 0...n) 0];
        var perm1 = [for (i in 0...n) i];
        var count = [for (i in 0...n) 0];
        var maxFlips = 0;
        var checksum = 0;
        var permCount = 0;
        var r = n;
        while (true) {
            while (r != 1) {
                count[r - 1] = r;
                r--;
            }
            for (i in 0...n)
                perm[i] = perm1[i];
            var flips = 0;
            var k = perm[0];
            while (k != 0) {
                var k2 = (k + 1) >> 1;
                for (i in 0...k2) {
                    var t = perm[i];
                    perm[i] = perm[k - i];
                    perm[k - i] = t;
                }
                flips++;
                k = perm[0];
            }
            if (flips > maxFlips)
                maxFlips = flips;
            checksum += (permCount % 2 == 0) ? flips : -flips;

            // Next permutation in the count-vector order.
            while (true) {
                if (r == n)
                    return [checksum, maxFlips];
                var perm0 = perm1[0];
                var i = 0;
                while (i < r) {
                    var j = i + 1;
                    perm1[i] = perm1[j];
                    i = j;
                }
                perm1[r] = perm0;
                count[r] = count[r] - 1;
                if (count[r] > 0)
                    break;
                r++;
            }
            permCount++;
        }
    }

    public static function main() {
        var res = fannkuch(N);
        Sys.println(res[0] + "\nPfannkuchen(" + N + ") = " + res[1]);
    }
}
