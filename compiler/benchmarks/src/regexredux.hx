// Regex-redux benchmark — the Computer Language Benchmarks Game algorithm,
// single-threaded. The FASTA input, normally on stdin, is produced in-process
// by the FASTA generator.
//
// Tests: EReg matching and replacement over a large string

package benchmarks;

class RegexRedux {
    static inline var N = 50000;
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

    static var seed = 42;

    static inline function random(max:Float):Float {
        seed = (seed * IA + IC) % IM;
        return max * seed / IM;
    }

    static function repeatFasta(sb:StringBuf, header:String, src:String, n:Int) {
        sb.add(header);
        var len = src.length;
        var pos = 0;
        var left = n;
        while (left > 0) {
            var line = left < LINE ? left : LINE;
            for (i in 0...line) {
                sb.addChar(src.charCodeAt(pos));
                pos++;
                if (pos == len)
                    pos = 0;
            }
            sb.addChar(10);
            left -= line;
        }
    }

    static function randomFasta(sb:StringBuf, header:String, codes:Array<Int>, probs:Array<Float>, n:Int) {
        sb.add(header);
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
                sb.addChar(codes[j]);
            }
            sb.addChar(10);
            left -= line;
        }
    }

    static function codesOf(s:String):Array<Int> {
        return [for (i in 0...s.length) s.charCodeAt(i)];
    }

    static function makeInput(n:Int):String {
        seed = 42;
        var sb = new StringBuf();
        repeatFasta(sb, ">ONE Homo sapiens alu\n", ALU, n * 2);
        randomFasta(sb, ">TWO IUB ambiguity codes\n", codesOf("acgtBDHKMNRSVWY"),
            [0.27, 0.12, 0.12, 0.27, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], n * 3);
        randomFasta(sb, ">THREE Homo sapiens frequency\n", codesOf("acgt"),
            [0.3029549426680, 0.1979883004921, 0.1975473066391, 0.3015094502008], n * 5);
        return sb.toString();
    }

    static function countMatches(re:EReg, s:String):Int {
        var count = 0;
        var pos = 0;
        while (pos <= s.length && re.matchSub(s, pos)) {
            count++;
            var m = re.matchedPos();
            pos = m.pos + (m.len > 0 ? m.len : 1);
        }
        return count;
    }

    public static function main() {
        var input = makeInput(N);
        var seq = new EReg(">.*\n|\n", "g").replace(input, "");
        var cleanLength = seq.length;

        var variants = [
            "agggtaaa|tttaccct",
            "[cgt]gggtaaa|tttaccc[acg]",
            "a[act]ggtaaa|tttacc[agt]t",
            "ag[act]gtaaa|tttac[agt]ct",
            "agg[act]taaa|ttta[agt]cct",
            "aggg[acg]aaa|ttt[cgt]ccct",
            "agggt[cgt]aa|tt[acg]accct",
            "agggta[cgt]a|t[acg]taccct",
            "agggtaa[cgt]|[acg]ttaccct"
        ];
        for (v in variants)
            Sys.println(v + " " + countMatches(new EReg(v, ""), seq));

        var subst = [
            ["tHa[Nt]", "<4>"],
            ["aND|caN|Ha[DS]|WaS", "<3>"],
            ["a[NSt]|BY", "<2>"],
            ["<[^>]*>", "|"],
            ["\\|[^|][^|]*\\|", "-"]
        ];
        for (s in subst)
            seq = new EReg(s[0], "g").replace(seq, s[1]);

        Sys.println("");
        Sys.println(input.length);
        Sys.println(cleanLength);
        Sys.println(seq.length);
    }
}
