import nue.loader.GGUFLoader;
import nue.sampling.ArgmaxSampler;
import nue.sampling.Sampler;
import nue.sampling.GenerationLoop;
import nue.tokenizer.Tokenizer;
import nue.CausalLanguageModel;
import rayzor.ds.Tensor;

/**
 * Local temperature sampler. Lifted out of `nue.sampling.TemperatureSampler`
 * to dodge a current JIT issue: importing that class causes a trap-stub
 * cascade in unrelated functions even when nothing instantiates it.
 * Under investigation; meanwhile this inline copy gets the demo
 * working with temperature-controlled sampling.
 */
class LocalTempSampler implements Sampler {
    public var temperature:Float;
    private var state:Int;

    public function new(temperature:Float, seed:Int) {
        this.temperature = temperature;
        this.state = seed;
    }

    public function sample(logits:Tensor):Int {
        var shape = logits.shape();
        var n = shape[shape.length - 1];
        var t = (temperature <= 0.0) ? 0.00000001 : temperature;

        var maxLogit = logits.get([0]);
        for (i in 1...n) {
            var v = logits.get([i]);
            if (v > maxLogit) maxLogit = v;
        }

        // Two-pass: accumulate `total` while computing exp(logit/t)
        // on the fly the SECOND time. Skips the Array<Float> buffer
        // entirely so we don't trip any precision/JIT bugs.
        var total = 0.0;
        for (i in 0...n) {
            total += Math.exp((logits.get([i]) - maxLogit) / t);
        }

        var r = nextFloat() * total;
        var acc = 0.0;
        for (i in 0...n) {
            acc += Math.exp((logits.get([i]) - maxLogit) / t);
            if (r <= acc) return i;
        }
        return n - 1;
    }

    private function nextFloat():Float {
        state = (state * 1664525 + 1013904223) & 0x7FFFFFFF;
        return state / 2147483648.0;
    }
}

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
    // Default context cap. GGUFs advertise up to 131072 (Llama 3.2 1B);
    // honouring that pre-allocates ~8.6 GB of empty KV cache per layer at
    // F32. 4096 is plenty for short prompts + generation and keeps the
    // KV cache around ~135 MB.
    static inline var DEFAULT_CTX = 4096;

    static function main() {
        var args = Sys.args();
        if (args.length < 1) {
            Sys.println("usage: rayzor run Main.hx -- <model.gguf> [prompt] [max_tokens] [ctx] [temperature]");
            Sys.exit(1);
        }
        var path = args[0];
        var prompt = (args.length > 1) ? args[1] : DEFAULT_PROMPT;
        var maxNewRaw = (args.length > 2) ? Std.parseInt(args[2]) : null;
        var maxNew:Int = DEFAULT_MAX_TOKENS;
        if (maxNewRaw != null) {
            var unboxed = maxNewRaw + 0;
            if (unboxed > 0) maxNew = unboxed;
        }
        var ctxRaw = (args.length > 3) ? Std.parseInt(args[3]) : null;
        var ctx:Int = DEFAULT_CTX;
        if (ctxRaw != null) {
            // `+ 0` forces an Int unbox of the Null<Int> result — assigning
            // `ctxRaw` to an Int local without an arithmetic op leaves the
            // boxed pointer in the register, which then propagates as
            // garbage across cross-file function calls.
            var unboxed = ctxRaw + 0;
            if (unboxed > 0) ctx = unboxed;
        }
        // Temperature for sampling. 0.0 → greedy (Argmax). 0.7 → standard
        // chat preset. Higher = more random. Default 0.0 keeps the run
        // deterministic; pass a non-zero value to break greedy loops.
        var temperature:Float = 0.0;
        if (args.length > 4) {
            var parsed = Std.parseFloat(args[4]);
            // Guard against NaN by comparing against itself (only NaN
            // != NaN). Negative temps clamp to 0 (greedy).
            if (parsed == parsed && parsed >= 0.0) temperature = parsed;
        }

        trace("=== nue llama-chat ===");
        trace("model:  " + path);
        trace("prompt: \"" + prompt + "\"");
        trace("max:    " + maxNew + " tokens");
        trace("ctx:    " + ctx + " tokens");
        trace("temp:   " + temperature);
        trace("");

        // One open of the file: header + tensor index parsed once,
        // model and tokenizer both materialise off the same reader.
        var loader = new GGUFLoader();
        trace("[load] reading GGUF (dequants Q4_K_M weights to F32)...");
        var startLoad = Sys.time();
        var loaded = loader.loadWithTokenizer(path, ctx);
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

        // Sampler choice:
        //   temp = 0.0 → Argmax (deterministic, greedy).
        //   temp > 0.0 → TemperatureSampler with softmax + multinomial
        //     draw. O(n) per token over the full vocab — TopP/TopK
        //     would add a sort over 128k tokens (Llama 3 vocab) which
        //     burns minutes per token until a partial-sort variant
        //     lands. Plain temperature scaling is the fast knob for
        //     breaking greedy-loop repetitions while keeping per-token
        //     cost in the same ballpark as argmax.
        var sampler:Sampler = (temperature > 0.0)
            ? new LocalTempSampler(temperature, 42)
            : new ArgmaxSampler();

        // Instruct models behave best when the prompt is wrapped in the
        // model's chat template. Llama-3 uses
        //   <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
        //   {prompt}<|eot_id|>
        //   <|start_header_id|>assistant<|end_header_id|>\n\n
        // The BPE tokenizer recognises registered specials as atomic
        // ids; GGUFTokenizer registers the Llama-3 chat specials by
        // direct vocab lookup so this string round-trips correctly.
        var startHdr = tok.specialId("<|start_header_id|>");
        var modelPrompt:String = prompt;
        if (startHdr >= 0) {
            modelPrompt =
                "<|begin_of_text|>"
                + "<|start_header_id|>user<|end_header_id|>\n\n"
                + prompt
                + "<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n\n";
            trace("[chat] wrapping prompt in Llama-3 Instruct template");
        }

        var loop = new GenerationLoop(model, tok, sampler, eos, maxNew);

        trace("[gen] streaming...");
        // Live print: emit the partial-text delta each step so the
        // user sees tokens appear in real time. The callback's
        // `partial` is the full decoded tail so far, not just the
        // newest piece, so we track the previous length and slice.
        var prevLen = 0;
        var startedAt = Sys.time();
        var nTokens = 0;
        var output = loop.generate(modelPrompt, function(_id:Int, partial:String):Bool {
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
