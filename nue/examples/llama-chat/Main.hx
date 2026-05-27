import nue.loader.GGUFLoader;
import nue.sampling.ArgmaxSampler;
import nue.sampling.GenerationLoop;
import nue.tokenizer.Tokenizer;
import nue.CausalLanguageModel;

/**
 * Phase 9 demo — load a real GGUF off disk, build the model + tokenizer,
 * and stream generated tokens via `GenerationLoop`.
 *
 * Usage:
 *   rayzor run Main.hx -- <model.gguf> [prompt="Hello"] [max_tokens=32]
 *
 * Memory caveat — Phase 4a only:
 *   GGUFLoader currently dequantises Q4_K_M tensors to F32 at load
 *   time (Phase 4a path) because `Linear` doesn't yet accept a
 *   `QTensor` weight. That's roughly an 8× expansion vs. on-disk:
 *     - Llama 3.2 1B Q4_K_M: ~770 MB on disk → ~5.5 GB resident
 *     - Llama 3   8B Q4_K_M: ~4.6 GB on disk → ~32 GB resident
 *   On a 16 GB machine, only the 1B (or smaller) will fit without
 *   thrashing swap. Phase 4b — `Linear` dispatching directly on
 *   `QTensor.matmulF32` — is the follow-up that keeps weights
 *   compressed in RAM.
 *
 * Status: wiring complete; readMetadata + tokenizer extraction work
 * on real Llama 3.2 1B Q4_K_M. Full weight decode crashes silently
 * during `tensorsFromReader` — under investigation.
 */
class Main {
    static inline var DEFAULT_PROMPT = "Hello";
    static inline var DEFAULT_MAX_TOKENS = 32;

    static function main() {
        var args = Sys.args();
        if (args.length < 1) {
            Sys.println("usage: rayzor run Main.hx -- <model.gguf> [prompt] [max_tokens]");
            Sys.exit(1);
        }
        var path = args[0];
        var prompt = (args.length > 1) ? args[1] : DEFAULT_PROMPT;
        var maxNew = (args.length > 2) ? Std.parseInt(args[2]) : DEFAULT_MAX_TOKENS;
        if (maxNew == null || maxNew <= 0) maxNew = DEFAULT_MAX_TOKENS;

        trace("=== nue llama-chat ===");
        trace("model:  " + path);
        trace("prompt: \"" + prompt + "\"");
        trace("max:    " + maxNew + " tokens");
        trace("");

        // One open of the file: header + tensor index parsed once,
        // model and tokenizer both materialise off the same reader.
        var loader = new GGUFLoader();
        trace("[load] reading GGUF (dequants Q4_K_M weights to F32)...");
        var startLoad = Sys.time();
        var loaded = loader.loadWithTokenizer(path);
        trace("[load] done in " + fmt(Sys.time() - startLoad) + "s");

        var meta = loaded.metadata;
        var tok = loaded.tokenizer;
        // Loader returns `Module`; `GenerationLoop` needs the causal
        // LM facet (forwardIds + resetCache). All `ArchBuilder`s
        // hand back a `CausalLanguageModel`, so the cast is safe.
        var model = cast(loaded.model, CausalLanguageModel);

        trace("[meta] " + meta.architecture + " hidden=" + meta.hiddenSize
            + " layers=" + meta.numLayers + " heads=" + meta.numHeads
            + "/" + meta.numKvHeads + " ffn=" + meta.intermediateSize
            + " vocab=" + meta.vocabSize + " ctx=" + meta.maxSeqLen);
        trace("[tok]  vocab=" + tok.vocabSize());

        // Llama 3 uses `<|end_of_text|>` for base and `<|eot_id|>` for
        // the instruct-chat turn boundary. Prefer the chat sentinel
        // when present (instruct GGUFs) — falls back to end_of_text
        // for base models, then -1 if neither exists (decoder runs
        // until `maxNew` tokens).
        var eos = tok.specialId("<|eot_id|>");
        if (eos < 0) eos = tok.specialId("<|end_of_text|>");
        trace("[tok]  eos=" + eos);
        trace("");

        var sampler = new ArgmaxSampler();
        var loop = new GenerationLoop(model, tok, sampler, eos, maxNew);

        trace("[gen] streaming...");
        // Live print: emit the partial-text delta each step so the
        // user sees tokens appear in real time. The callback's
        // `partial` is the full decoded tail so far, not just the
        // newest piece, so we track the previous length and slice.
        var prevLen = 0;
        var startedAt = Sys.time();
        var nTokens = 0;
        var output = loop.generate(prompt, function(_id:Int, partial:String):Bool {
            var delta = partial.substr(prevLen);
            prevLen = partial.length;
            Sys.print(delta);
            nTokens++;
            return true;
        });
        var elapsed = Sys.time() - startedAt;
        Sys.println("");
        trace("");
        trace("[done] " + nTokens + " tokens in " + fmt(elapsed) + "s ("
            + fmt(nTokens / elapsed) + " tok/s)");
        trace("[output] " + output);
    }

    static inline function fmt(x:Float):String {
        return Std.string(Math.round(x * 1000) / 1000);
    }
}
