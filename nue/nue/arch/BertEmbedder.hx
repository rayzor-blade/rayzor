package nue.arch;

import sys.io.File;
import nue.loader.GGUFLoader;
import rayzor.ds.Tensor;

/**
 * Sentence-embedding driver. ALL BertModel construction and field access
 * happens inside this (nue.arch) module — the same module BertModel is
 * defined in. Callers in other modules (examples, servers) invoke only
 * STATIC methods with primitive args/returns; they never `new` a
 * nue.arch object nor read its fields across a module boundary, both of
 * which corrupt the field layout (bugs_import_xmodule_member_resolution).
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
}
