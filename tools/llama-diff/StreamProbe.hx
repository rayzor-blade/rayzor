import nue.tokenizer.BPETokenizer;
import nue.tokenizer.Vocab;
import nue.tokenizer.MergeRule;

/** Minimal streaming-decode probe: synthetic 5-token vocab, no GGUF/loader —
    compiles in seconds. Fast regression check for the tokenizer decode path
    (expected stream total: 'The is Paris.'). */
class StreamProbe {
    static function main() {
        var v = new Vocab();
        v.add("The");        // id 0
        v.add("ĠParis");     // id 1 (byte-alphabet space prefix)
        v.add("Ġis");        // id 2
        v.add(".");          // id 3
        var merges:Array<MergeRule> = [];
        var bpe = new BPETokenizer(v, merges, true);

        Sys.println("[sp] decode([0,2,1,3])  = '" + bpe.decode([0, 2, 1, 3]) + "'");
        Sys.println("[sp] decodePiece(1)     = '" + bpe.decodePiece(1) + "'");

        var carry:Array<haxe.io.Bytes> = [haxe.io.Bytes.alloc(0)];
        var out = "";
        for (id in [0, 2, 1, 3]) {
            var d = bpe.decodeStreamStep(carry, id);
            Sys.println("[sp] step(" + id + ") delta='" + d + "' len=" + d.length);
            out += d;
        }
        Sys.println("[sp] stream total       = '" + out + "' (expect 'The is Paris.')");
    }
}
