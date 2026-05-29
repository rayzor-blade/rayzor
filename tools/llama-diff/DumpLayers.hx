import nue.loader.GGUFLoader;
import nue.arch.LlamaModel;
import nue.transformer.TransformerBlock;
import nue.transformer.RMSNorm;
import nue.transformer.GQAttention;
import nue.Linear;
import rayzor.ds.Tensor;

/**
 * Diagnostic harness that mirrors llama.cpp's `llama-eval-callback`
 * output for Rayzor's Llama forward pass. Loads the GGUF, encodes a
 * fixed prompt (matching what llama.cpp will see for the same input),
 * then walks the model layer by layer printing `sum()` of each
 * intermediate tensor.
 *
 * The output format is `[sum] <label> = <value>`. The companion shell
 * harness greps these out and diffs them against the llama.cpp side
 * to find the first divergent operation.
 *
 * Usage:
 *   rayzor run DumpLayers.hx -- <model.gguf> "<prompt>"
 *
 * Default prompt is "Hello" (tokens [128000, 9906]) — same as the
 * `llama-eval-callback` reference command:
 *
 *   llama-eval-callback -m model.gguf -p "Hello" -n 1 --temp 0
 */
class DumpLayers {
    static function main() {
        var args = Sys.args();
        if (args.length < 1) {
            Sys.println("usage: rayzor run DumpLayers.hx -- <model.gguf> [prompt='Hello']");
            Sys.exit(1);
        }
        var path = args[0];
        var prompt = (args.length > 1) ? args[1] : "Hello";

        var loader = new GGUFLoader();
        var loaded = loader.loadWithTokenizer(path, 256);
        var tok = loaded.tokenizer;
        var model = cast(loaded.model, LlamaModel);

        // Encode the prompt. For "Hello" this should produce
        // [128000 (<|begin_of_text|>), 9906 (Hello)] matching what
        // `llama-eval-callback -p "Hello"` runs.
        var ids = tok.encode(prompt);
        // llama.cpp prepends BOS automatically when invoked with -p.
        // Our encoder does NOT, so prepend manually so the dumps line
        // up across both pipelines.
        var bos = tok.specialId("<|begin_of_text|>");
        if (bos >= 0 && (ids.length == 0 || ids[0] != bos)) {
            ids.unshift(bos);
        }

        Sys.print("[ids] ");
        for (i in 0...ids.length) Sys.print(ids[i] + " ");
        Sys.println("");

        // ---- Forward pass with per-tensor sum dumps ----

        var x = model.embedTokens.lookup(ids);
        dump("embd", x);

        for (i in 0...model.blocks.length) {
            var blk = cast(model.blocks[i], TransformerBlock);
            var attnNorm = cast(blk.attnNorm, RMSNorm);
            var ffnNorm = cast(blk.ffnNorm, RMSNorm);
            var attn = cast(blk.attn, GQAttention);

            // Attn sub-layer
            var normed = attnNorm.forward(x);
            dump("attn_norm-" + i, normed);

            // Q / K / V projections
            var qProj = attn.qProj;
            var kProj = attn.kProj;
            var vProj = attn.vProj;
            var q = qProj.forward(normed);
            var k = kProj.forward(normed);
            var v = vProj.forward(normed);
            dump("Qcur-" + i, q);
            dump("Kcur-" + i, k);
            dump("Vcur-" + i, v);

            // Drill into layer 0 attention: manually apply RoPE then
            // dump the result so we can compare against llama.cpp's
            // `Qcur (ROPE)` / `Kcur (ROPE)` sums and isolate where
            // the attention path first diverges.
            if (i == 0) {
                var seqQ = q.shape()[0];
                var qReshaped = q.reshape([seqQ, attn.numQHeads, attn.headDim]);
                var kReshaped = k.reshape([seqQ, attn.numKvHeads, attn.headDim]);
                var qRot = attn.rope.apply(qReshaped, attn.cache.currentLen);
                var kRot = attn.rope.apply(kReshaped, attn.cache.currentLen);
                dump("Qcur-" + i + "-rope", qRot);
                dump("Kcur-" + i + "-rope", kRot);
            }

            // Full attn forward (does RoPE + cache + attn + O proj
            // internally; we don't tap into the intermediates here
            // because GQAttention doesn't currently expose them).
            var attnOut = attn.forward(normed);
            dump("attn_out-" + i, attnOut);

            // Residual + FFN sub-layer
            var h1 = x.add(attnOut);
            dump("ffn_inp-" + i, h1);

            var ffnNormed = ffnNorm.forward(h1);
            dump("ffn_norm-" + i, ffnNormed);

            var ffnOut = blk.ffn.forward(ffnNormed);
            dump("ffn_out-" + i, ffnOut);

            x = h1.add(ffnOut);
            dump("l_out-" + i, x);
        }

        // Final RMSNorm + LM head
        var normed = model.outputNorm.forward(x);
        dump("output_norm", normed);

        var logits = model.lmHead.forward(normed);
        dump("logits", logits);

        Sys.println("[done]");
    }

    static inline function dump(label:String, t:Tensor):Void {
        Sys.println("[sum] " + label + " = " + fmt(t.sum()));
    }

    static inline function fmt(x:Float):String {
        // 6 decimals matches llama.cpp's `sum = N.NNNNNN` format.
        var scaled = Math.round(x * 1000000) / 1000000;
        return Std.string(scaled);
    }
}
