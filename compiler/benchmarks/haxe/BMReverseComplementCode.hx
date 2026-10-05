// Reverse-complement benchmark — the Computer Language Benchmarks Game
// algorithm. The FASTA input, normally on stdin, is produced in-process by the
// FASTA generator; the output is hashed rather than written to stdout.
//
// Tests: string scanning, lookup tables, StringBuf building, reversal

class BMReverseComplementCode {
    static inline var N = 250000;
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

    static function complementTable():Array<Int> {
        var table = [for (i in 0...128) i];
        var from = "ACBDGHKMNSRUTWVYacbdghkmnsrutwvy";
        var to = "TGVHCDMKNSYAAWBRTGVHCDMKNSYAAWBR";
        for (i in 0...from.length)
            table[from.charCodeAt(i)] = to.charCodeAt(i);
        return table;
    }

    var table:Array<Int>;
    var hash:Int;
    var emitted:Int;

    function new() {
        table = complementTable();
        hash = 0;
        emitted = 0;
    }

    inline function emit(c:Int) {
        hash = (hash * 31 + c) & 0xFFFFFF;
        emitted++;
    }

    // Writes the reverse complement of `seq[0...len]` in lines of LINE.
    function flush(seq:Array<Int>, len:Int) {
        var col = 0;
        var i = len - 1;
        while (i >= 0) {
            emit(table[seq[i]]);
            col++;
            if (col == LINE) {
                emit(10);
                col = 0;
            }
            i--;
        }
        if (col > 0)
            emit(10);
    }

    function process(input:String) {
        var seq:Array<Int> = [];
        var len = 0;
        var i = 0;
        var n = input.length;
        while (i < n) {
            var c = input.charCodeAt(i);
            if (c == 62) {
                flush(seq, len);
                len = 0;
                while (i < n && input.charCodeAt(i) != 10) {
                    emit(input.charCodeAt(i));
                    i++;
                }
                emit(10);
            } else if (c != 10) {
                if (len < seq.length)
                    seq[len] = c;
                else
                    seq.push(c);
                len++;
            }
            i++;
        }
        flush(seq, len);
    }

    public static function main() {
        var input = makeInput(N);
        var rc = new BMReverseComplementCode();
        rc.process(input);
        Sys.println("bytes " + rc.emitted + " hash " + rc.hash);
    }
}
