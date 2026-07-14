package nue.arch;

import sys.io.File;
import nue.loader.GGUFLoader;
import nue.loader.GGUFReader;
import nue.loader.GGUFTokenizer;
import rayzor.ds.Tensor;

/**
 * Sentence-embedding driver. BertModel construction and field access stay
 * inside nue.arch; callers in other modules use the static entry points,
 * which take and return primitives.
 */
class BertEmbedder {
    var model:BertModel;
    var dim:Int;

    function new(ggufPath:String) {
        var loader = new GGUFLoader();
        // Two-arg CHECKED cast — the one-arg `cast x` unsafe form does not
        // recover the concrete object layout from a Module return, so field
        // reads come back zero. llama-chat uses cast(loaded.model, LlamaModel)
        // for the same reason.
        this.model = cast(loader.load(ggufPath), BertModel);
        this.dim = model.meta.hiddenSize;
    }

    /** Encode + mean-pool + L2, as a plain float array. */
    function embed(ids:Array<Int>):Array<Float> {
        var e:Tensor = model.embed(ids);
        var out = [for (j in 0...dim) e.getFlat(j)];
        e.free();
        return out;
    }

    /**
     * Golden correctness test — entirely in-module. Reads the id fixture
     * and the f32 embedding golden, embeds each sentence, prints per-row
     * cosine, and the min/mean. Returns the min cosine.
     */
    public static function goldenTest(ggufPath:String, idsPath:String, goldPath:String):Float {
        var self = new BertEmbedder(ggufPath);
        Sys.println("model loaded, dim=" + self.dim);

        var idLines = File.getContent(idsPath).split("\n");
        var gold = File.getBytes(goldPath);
        var dim = self.dim;

        var minCos = 2.0;
        var sumCos = 0.0;
        var rows = 0;
        for (r in 0...idLines.length) {
            var line = StringTools.trim(idLines[r]);
            if (line == "") continue;
            var ids = [for (p in line.split(" ")) Std.parseInt(p)];

            var emb = self.embed(ids);
            var dot = 0.0;
            var na = 0.0;
            var nb = 0.0;
            for (j in 0...dim) {
                var a = emb[j];
                var b = gold.getFloat((r * dim + j) * 4);
                dot += a * b;
                na += a * a;
                nb += b * b;
            }
            var cos = dot / (Math.sqrt(na) * Math.sqrt(nb));
            if (cos < minCos) minCos = cos;
            sumCos += cos;
            rows++;
            Sys.println("row " + r + "  cos=" + cos + "  (n_tok=" + ids.length + ")");
        }
        Sys.println("=== cosine  min=" + minCos + "  mean=" + (sumCos / rows)
            + "  over " + rows + " rows   gate>=0.999 ===");
        return minCos;
    }

    /**
     * WordPiece tokenizer dump — encode each sentence to its `[CLS]…[SEP]` id
     * sequence and print one `IDS <row> <id>...` line per sentence. The caller
     * compares those against the golden ids externally.
     */
    public static function tokenizerTest(ggufPath:String, sentencesPath:String, expectedIdsPath:String):Int {
        var bytes = File.getBytes(ggufPath);
        var reader = new GGUFReader(bytes);
        var tok = GGUFTokenizer.buildWordPiece(reader);
        Sys.println("tokenizer built, vocab=" + tok.vocabSize());

        // Emit one `IDSV <row> <id>...` line per sentence; the caller diffs
        // these against the golden ids (harness_ids.txt). Kept as a dump + an
        // external compare because an in-Haxe comparison hits an unrelated
        // native codegen crash (see bugs_native_codegen_continue_and_loop_output).
        var sents = File.getContent(sentencesPath).split("\n");
        for (i in 0...sents.length) {
            var s = StringTools.trim(sents[i]);
            if (s.length == 0) continue;
            var got = tok.encodeWithSpecials(s);
            var line = "" + i;
            for (k in 0...got.length) line += " " + got[k];
            Sys.println("IDSV " + line);
        }
        return 0;
    }
}
