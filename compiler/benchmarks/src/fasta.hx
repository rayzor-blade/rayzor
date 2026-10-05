// FASTA benchmark — the Computer Language Benchmarks Game algorithm. The
// generated text is hashed as it is produced instead of written to stdout, so
// the kernel measures generation rather than terminal I/O.
//
// Tests: LCG arithmetic, cumulative-probability lookup, byte emission

package benchmarks;

class Fasta {
    static inline var N = 2500000;
    static inline var LINE = 60;
    static inline var IM = 139968;
    static inline var IA = 3877;
    static inline var IC = 29573;

    static inline var ALU = "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG"
        + "GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA"
        + "CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT"
        + "ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA"
        + "GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG"
        + "AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC"
        + "AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";

    var seed:Int;
    var hash:Int;
    var emitted:Int;

    public function new() {
        seed = 42;
        hash = 0;
        emitted = 0;
    }

    inline function random(max:Float):Float {
        seed = (seed * IA + IC) % IM;
        return max * seed / IM;
    }

    inline function emit(c:Int) {
        hash = (hash * 31 + c) & 0xFFFFFF;
        emitted++;
    }

    function emitString(s:String) {
        for (i in 0...s.length)
            emit(s.charCodeAt(i));
    }

    function repeatFasta(header:String, src:String, n:Int) {
        emitString(header);
        var codes = codesOf(src);
        var len = codes.length;
        var pos = 0;
        var left = n;
        while (left > 0) {
            var line = left < LINE ? left : LINE;
            for (i in 0...line) {
                emit(codes[pos]);
                pos++;
                if (pos == len)
                    pos = 0;
            }
            emit(10);
            left -= line;
        }
    }

    function randomFasta(header:String, codes:Array<Int>, probs:Array<Float>, n:Int) {
        emitString(header);
        var cum = [];
        var acc = 0.0;
        for (p in probs) {
            acc += p;
            cum.push(acc);
        }
        var last = codes.length - 1;
        var left = n;
        while (left > 0) {
            var line = left < LINE ? left : LINE;
            for (i in 0...line) {
                var r = random(1.0);
                var j = 0;
                while (j < last && r >= cum[j])
                    j++;
                emit(codes[j]);
            }
            emit(10);
            left -= line;
        }
    }

    static function codesOf(s:String):Array<Int> {
        return [for (i in 0...s.length) s.charCodeAt(i)];
    }

    public static function main() {
        var f = new Fasta();
        f.repeatFasta(">ONE Homo sapiens alu\n", ALU, N * 2);
        f.randomFasta(">TWO IUB ambiguity codes\n", codesOf("acgtBDHKMNRSVWY"),
            [0.27, 0.12, 0.12, 0.27, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], N * 3);
        f.randomFasta(">THREE Homo sapiens frequency\n", codesOf("acgt"),
            [0.3029549426680, 0.1979883004921, 0.1975473066391, 0.3015094502008], N * 5);
        Sys.println("bytes " + f.emitted + " hash " + f.hash);
    }
}
