package nue.arch;

import sys.io.File;
import nue.loader.GGUFLoader;
import nue.loader.GGUFReader;
import nue.loader.GGUFTokenizer;
import nue.tokenizer.WordPieceTokenizer;
import rayzor.ds.Tensor;

/**
 * Sentence-embedding driver: text → WordPiece → encoder → mean-pool + L2.
 * BertModel construction and field access stay inside nue.arch; callers in
 * other modules use the static entry points, which take and return primitives.
 */
class BertEmbedder {
    var model:BertModel;
    var tok:WordPieceTokenizer;
    var dim:Int;

    function new(ggufPath:String) {
        var loader = new GGUFLoader();
        // Two-arg CHECKED cast — the one-arg `cast x` unsafe form does not
        // recover the concrete object layout from a Module return, so field
        // reads come back zero. llama-chat uses cast(loaded.model, LlamaModel)
        // for the same reason.
        this.model = cast(loader.load(ggufPath), BertModel);
        this.dim = model.meta.hiddenSize;
        this.tok = GGUFTokenizer.buildWordPiece(new GGUFReader(File.getBytes(ggufPath)));
    }

    /** Encode + mean-pool + L2, as a plain float array. */
    function embed(ids:Array<Int>):Array<Float> {
        var e:Tensor = model.embed(ids);
        var out = [for (j in 0...dim) e.getFlat(j)];
        e.free();
        return out;
    }

    /** Full pipeline: raw text → WordPiece ([CLS]…[SEP]) → encode → pool + L2. */
    function embedText(text:String):Array<Float> {
        return embed(tok.encodeWithSpecials(text));
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
     * END-TO-END correctness: embed each sentence from raw TEXT (tokenize +
     * encode + pool) and compare to the sentence-transformers golden embedding.
     * No golden ids are fed in — this exercises the whole pipeline. Returns the
     * min cosine.
     */
    public static function textGoldenTest(ggufPath:String, sentencesPath:String, goldPath:String):Float {
        var self = new BertEmbedder(ggufPath);
        Sys.println("model+tokenizer loaded, dim=" + self.dim + " vocab=" + self.tok.vocabSize());

        var sents = File.getContent(sentencesPath).split("\n");
        var gold = File.getBytes(goldPath);
        var dim = self.dim;

        var minCos = 2.0;
        var sumCos = 0.0;
        var rows = 0;
        for (r in 0...sents.length) {
            var s = StringTools.trim(sents[r]);
            if (s.length == 0) continue;
            var emb = self.embedText(s);
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
            Sys.println("row " + r + "  cos=" + cos);
        }
        Sys.println("=== TEXT→embedding cosine  min=" + minCos + "  mean=" + (sumCos / rows)
            + "  over " + rows + " rows   gate>=0.999 ===");
        return minCos;
    }

    /** `nue embed`: load, embed one text, print the dim, L2 norm, and head. */
    public static function embedOne(ggufPath:String, text:String):Void {
        var self = new BertEmbedder(ggufPath);
        var emb = self.embedText(text);
        var norm = 0.0;
        for (v in emb) norm += v * v;
        var head = "";
        for (j in 0...8) head += " " + emb[j];
        Sys.println("text: \"" + text + "\"");
        Sys.println("embedding dim=" + self.dim + "  L2=" + Math.sqrt(norm) + "  head:" + head);
    }

    /**
     * WordPiece tokenizer correctness: encode each sentence to its `[CLS]…[SEP]`
     * id sequence and compare against the golden ids (exact match). Prints the
     * pass/total and returns the number of mismatched sentences.
     */
    public static function tokenizerTest(ggufPath:String, sentencesPath:String, expectedIdsPath:String):Int {
        var bytes = File.getBytes(ggufPath);
        var reader = new GGUFReader(bytes);
        var tok = GGUFTokenizer.buildWordPiece(reader);
        Sys.println("tokenizer built, vocab=" + tok.vocabSize());

        var sents = File.getContent(sentencesPath).split("\n");
        var expLines = File.getContent(expectedIdsPath).split("\n");
        var pass = 0;
        var fail = 0;
        for (i in 0...sents.length) {
            var s = StringTools.trim(sents[i]);
            if (s.length == 0) continue;
            var got = tok.encodeWithSpecials(s);
            var exp = [for (p in StringTools.trim(expLines[i]).split(" ")) Std.parseInt(p)];
            var ok = got.length == exp.length;
            var j = 0;
            while (ok && j < got.length) {
                if (got[j] != exp[j]) ok = false;
                j++;
            }
            if (ok) {
                pass++;
            } else {
                fail++;
                var gl = "";
                for (k in 0...got.length) gl += " " + got[k];
                Sys.println("MISMATCH row " + i + " \"" + s + "\" got:" + gl);
            }
        }
        Sys.println("=== tokenizer: " + pass + "/" + (pass + fail) + " sentences exact-id match ===");
        return fail;
    }
}
