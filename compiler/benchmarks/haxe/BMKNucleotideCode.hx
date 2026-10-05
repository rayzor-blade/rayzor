// K-nucleotide benchmark — the Computer Language Benchmarks Game algorithm,
// single-threaded. The input, normally FASTA on stdin, is produced in-process
// by the FASTA generator; the THREE section is analysed. Counts are printed
// as integers, since float formatting differs between targets.
//
// Tests: substring extraction, Map<String, Int> hashing, sorting

class BMKNucleotideCode {
    static inline var N = 100000;
    static inline var IM = 139968;
    static inline var IA = 3877;
    static inline var IC = 29573;

    static var seed = 42;

    static inline function random(max:Float):Float {
        seed = (seed * IA + IC) % IM;
        return max * seed / IM;
    }

    static function randomSequence(codes:Array<Int>, probs:Array<Float>, n:Int):String {
        var cum = [];
        var acc = 0.0;
        for (p in probs) {
            acc += p;
            cum.push(acc);
        }
        var last = codes.length - 1;
        var sb = new StringBuf();
        for (i in 0...n) {
            var r = random(1.0);
            var j = 0;
            while (j < last && r >= cum[j])
                j++;
            sb.addChar(codes[j]);
        }
        return sb.toString();
    }

    static function codesOf(s:String):Array<Int> {
        return [for (i in 0...s.length) s.charCodeAt(i)];
    }

    // The ONE and TWO sections only advance the generator, as reading past
    // them in a FASTA file would; THREE is the sequence analysed.
    static function makeInput(n:Int):String {
        seed = 42;
        randomSequence(codesOf("acgtBDHKMNRSVWY"),
            [0.27, 0.12, 0.12, 0.27, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], n * 3);
        return randomSequence(codesOf("acgt"),
            [0.3029549426680, 0.1979883004921, 0.1975473066391, 0.3015094502008], n * 5).toUpperCase();
    }

    static function frequencies(seq:String, k:Int):Map<String, Int> {
        var counts = new Map<String, Int>();
        var end = seq.length - k + 1;
        for (i in 0...end) {
            var key = seq.substr(i, k);
            var c = counts.get(key);
            counts.set(key, c == null ? 1 : c + 1);
        }
        return counts;
    }

    static function sortedFrequencies(seq:String, k:Int):String {
        var counts = frequencies(seq, k);
        var keys = [for (key in counts.keys()) key];
        keys.sort(function(a, b) {
            var d = counts.get(b) - counts.get(a);
            if (d != 0)
                return d;
            return a < b ? -1 : (a > b ? 1 : 0);
        });
        var sb = new StringBuf();
        for (key in keys) {
            sb.add(key);
            sb.add(" ");
            sb.add(counts.get(key));
            sb.add("\n");
        }
        return sb.toString();
    }

    static function specificCount(seq:String, fragment:String):String {
        var counts = frequencies(seq, fragment.length);
        var c = counts.get(fragment);
        return (c == null ? 0 : c) + "\t" + fragment;
    }

    public static function main() {
        var seq = makeInput(N);
        Sys.print(sortedFrequencies(seq, 1));
        Sys.print(sortedFrequencies(seq, 2));
        for (fragment in ["GGT", "GGTA", "GGTATT", "GGTATTTTAATT", "GGTATTTTAATTTATAGT"])
            Sys.println(specificCount(seq, fragment));
    }
}
