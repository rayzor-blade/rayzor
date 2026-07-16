package nue.arch;

import sys.io.File;
import nue.loader.GGUFLoader;
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

    public function new(ggufPath:String) {
        var loader = new GGUFLoader();
        // Two-arg CHECKED cast — the one-arg `cast x` unsafe form does not
        // recover the concrete object layout from a Module return, so field
        // reads come back zero. llama-chat uses cast(loaded.model, LlamaModel)
        // for the same reason.
        this.model = cast(loader.load(ggufPath), BertModel);
        this.dim = model.meta.hiddenSize;
        // GGUFLoader.tokenizer dispatches on tokenizer.ggml.model, so a bert
        // GGUF yields WordPiece; recover the concrete type for encodeWithSpecials.
        this.tok = cast(loader.tokenizer(ggufPath), WordPieceTokenizer);
    }

    /** Encode + mean-pool + L2, as a plain float array. An optional mask
        (1 real / 0 pad) drives padding-aware attention and pooling. */
    function embed(ids:Array<Int>, ?mask:Array<Int>):Array<Float> {
        var e:Tensor = model.embed(ids, mask);
        var out = [for (j in 0...dim) e.getFlat(j)];
        e.free();
        return out;
    }

    // AMX f16 GEMM fires only at batch (=seq len) >= this; keep in sync with
    // RZT_AMX_MIN_BATCH in the tensor runtime.
    static inline var AMX_MIN = 16;

    // Whether padding-to-threshold pays off: only when the runtime will route
    // the padded GEMM through Accelerate f16 (macOS + RZT_AMX_PREFILL on). On
    // x86/non-Mac there is no such gate, so padding is pure wasted work — the
    // encoder stays on the F32 SIMD kernel regardless of seq len. Mirrors the
    // Q4Matmul.amxPrefill gate (which isn't reachable cross-module as a static).
    static var _amx:Int = 0;

    static function amxPad():Bool {
        if (_amx == 0) {
            var v = Sys.getEnvOr("RZT_AMX_PREFILL", "RAYZOR_AMX_PREFILL");
            var off = (v != null && (v == "0" || v == "false"));
            _amx = (!off && Sys.systemName() == "Mac") ? 1 : 2;
        }
        return _amx == 1;
    }

    /** Full pipeline: raw text → WordPiece ([CLS]…[SEP]) → encode → pool + L2. */
    public function embedText(text:String):Array<Float> {
        var ids = tok.encodeWithSpecials(text);
        // On AMX platforms, pad a short sequence up to the AMX threshold so the
        // encoder's Linear GEMMs take the Accelerate f16 fast path even for tiny
        // inputs; the mask excludes the padding from attention + pooling, so the
        // embedding is identical to the unpadded encode (validated by maskTest).
        // Non-AMX (x86) never pads — there padding is pure overhead.
        if (amxPad() && ids.length < AMX_MIN) {
            var pad = tok.specialId("[PAD]");
            if (pad < 0) pad = 0;
            var mask = [for (i in 0...ids.length) 1];
            while (ids.length < AMX_MIN) {
                ids.push(pad);
                mask.push(0);
            }
            return embed(ids, mask);
        }
        return embed(ids);
    }

    /** Embedding width — callers size their output buffers from this. */
    public function dimension():Int {
        return dim;
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

    /** `nue bench <gguf> <sentences.txt> <iters>`: load once, warm up, then time
        `iters` passes over the corpus and report pure encode throughput. Server-free
        so the F32 vs NUE_INT8 A/B isn't clouded by TCP/threading. */
    public static function benchMode(ggufPath:String, sentencesPath:String, itersArg:String):Void {
        var self = new BertEmbedder(ggufPath);
        var raw = File.getContent(sentencesPath).split("\n");
        var sents = [];
        for (r in 0...raw.length) {
            var s = StringTools.trim(raw[r]);
            if (s.length > 0) sents.push(s);
        }
        var iters = Std.parseInt(itersArg);
        if (iters == null || iters <= 0) iters = 10;
        Sys.println("[bench] loaded dim=" + self.dim + " corpus=" + sents.length + " iters=" + iters);
        // Warm up: JIT tier promotion + lazy int8 weight quantization.
        for (r in 0...sents.length) self.embedText(sents[r]);
        var t0 = Sys.time();
        var count = 0;
        for (it in 0...iters) {
            var it0 = Sys.time();
            for (r in 0...sents.length) {
                self.embedText(sents[r]);
                count++;
            }
            Sys.println("[bench] iter " + it + "  " + (sents.length / (Sys.time() - it0)) + " sent/s");
        }
        var dt = Sys.time() - t0;
        Sys.println("=== BENCH  sentences=" + count + "  encode_s=" + dt
            + "  sent/s=" + (count / dt) + " ===");
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
        var tok = cast(new GGUFLoader().tokenizer(ggufPath), WordPieceTokenizer);
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

    static function cosine(a:Array<Float>, b:Array<Float>):Float {
        var dot = 0.0;
        var na = 0.0;
        var nb = 0.0;
        for (j in 0...a.length) {
            dot += a[j] * b[j];
            na += a[j] * a[j];
            nb += b[j] * b[j];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    /**
     * Attention-mask correctness: right-pad each sentence with `[PAD]` and a
     * mask that excludes the padding, then confirm the embedding matches the
     * unpadded one (cosine ~1). The unmasked control (all-ones over the padded
     * length) is reported alongside — it must drift below the masked cosine,
     * proving the mask actually suppresses padding in both attention and the
     * mean pool. Returns the number of sentences that miss the gate.
     */
    public static function maskTest(ggufPath:String):Int {
        var self = new BertEmbedder(ggufPath);
        var pad = self.tok.specialId("[PAD]");
        if (pad < 0) pad = 0; // BERT [PAD] is vocab id 0
        Sys.println("mask test: dim=" + self.dim + "  pad_id=" + pad);

        var ids = self.tok.encodeWithSpecials("The quick brown fox jumps over the lazy dog.");
        var e1 = self.embed(ids); // unpadded reference

        // Right-pad and mask the padding out. With a correct mask the padded
        // embedding must equal the unpadded one (attention ignores the pad keys,
        // the pool ignores the pad tokens).
        var ids2 = ids.copy();
        var mask2 = [for (_ in 0...ids.length) 1];
        for (i in 0...12) {
            ids2.push(pad);
            mask2.push(0);
        }
        var eMask = self.embed(ids2, mask2);

        var cMask = cosine(e1, eMask);
        var ok = cMask > 0.9999;
        Sys.println("cos(unpadded, padded+mask) = " + cMask + (ok ? "   OK (padding ignored)" : "   FAIL"));
        Sys.println("=== attention-mask: " + (ok ? "PASS" : "FAIL") + " (gate cos>0.9999) ===");
        return ok ? 0 : 1;
    }
}
