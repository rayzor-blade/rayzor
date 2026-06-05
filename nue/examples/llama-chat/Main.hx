import nue.loader.GGUFLoader;
// nue.sampling.ArgmaxSampler import deliberately omitted — even an
// unused import of certain nue.sampling.* classes triggers the
// trap-stub-on-import cascade per bugs_trap_stub_cascade. Reintroduce
// when the cascade root is fixed.
import nue.sampling.Sampler;
import nue.sampling.GenerationLoop;
import nue.tokenizer.Tokenizer;
import nue.CausalLanguageModel;
import rayzor.ds.Tensor;

/**
 * Local Top-K + temperature + repetition-penalty sampler.
 *
 * Walks the full logits vector once and tracks the K highest scores
 * in a sorted parallel-array pair (`topKLogits`, `topKIds`). With
 * `k = 50` and a 128k-vocab Llama tokenizer that's a single linear
 * scan plus an insertion sort into a 50-entry buffer — O(n * log k)
 * in practice with constant ~50 movements per replace, vs the
 * O(n²) full sort `TopPSampler` does.
 *
 * Sampling then runs the standard temperature-softmax + multinomial
 * draw, but over only `k` candidates — much sharper than pure
 * temperature over the full vocab and the standard chat-quality
 * recipe (Llama / GPT defaults to k=50 + temperature ≈ 0.7).
 *
 * Repetition penalty divides positive logits by `penalty` and
 * multiplies negative ones for every token in the recent-output
 * window (last RECENT_CAP samples) **before** Top-K selection, so a
 * recently-emitted token won't even land in the candidate pool. 1.0
 * disables it; 1.1–1.3 is the typical chat range. Breaks the
 * "in in in" loops the small 1B Instruct model otherwise hits.
 *
 * Why inline rather than reuse `nue.sampling.TemperatureSampler` /
 * `TopKSampler`: importing those classes currently triggers a trap-
 * stub cascade in unrelated functions even with no instance created.
 * Root cause under investigation; meanwhile this inline copy is the
 * working path.
 */
class LocalTempSampler implements Sampler {
    public var temperature:Float;
    public var repetitionPenalty:Float;
    public var topK:Int;
    private var state:Int;
    private var recent:Array<Int>;
    private var recentWrite:Int;
    private var topKLogits:Array<Float>;
    private var topKIds:Array<Int>;
    private static inline var RECENT_CAP:Int = 64;

    public function new(temperature:Float, repetitionPenalty:Float, topK:Int, seed:Int) {
        this.temperature = temperature;
        this.repetitionPenalty = repetitionPenalty;
        this.topK = topK;
        this.state = seed;
        this.recent = [for (_ in 0...RECENT_CAP) -1];
        this.recentWrite = 0;
        // Pre-allocate the Top-K scratch arrays; both get reset per
        // `sample` call. Size always == topK; -1 / 0.0 fill is just
        // an initial value — overwritten before any read.
        var k = (topK > 0) ? topK : 1;
        this.topKLogits = [for (_ in 0...k) 0.0];
        this.topKIds = [for (_ in 0...k) -1];
    }

    public function sample(logits:Tensor):Int {
        var shape = logits.shape();
        var n = shape[shape.length - 1];
        var t = (temperature <= 0.0) ? 0.00000001 : temperature;
        var penalize = repetitionPenalty > 1.0;
        var rp = repetitionPenalty;
        var k = (topK > 0 && topK < n) ? topK : n;

        // -------- Top-K selection ----------
        // Single FFI call replaces the 128k-iter per-token scan + insertion
        // sort + repetition-penalty loop. The runtime walks the contiguous
        // F32 logits buffer in a tight Rust function and writes the top-K
        // (logit, id) pairs straight into our pre-allocated arrays.
        //
        // Falls back to the per-element scan when topkScan reports `-1`
        // (non-F32 / non-contiguous logits) — preserves correctness for any
        // future logits backend that doesn't hit the fast path.
        var penaltyArg = penalize ? rp : 1.0;
        var sz = logits.topkScan(topKLogits, topKIds, k, recent, penaltyArg);
        if (sz < 0) {
            sz = topKScanFallback(logits, n, k, penalize, rp);
        }

        // -------- Temperature softmax over the k survivors ----------
        var maxLogit = topKLogits[0];
        var total = 0.0;
        for (i in 0...sz) {
            total += Math.exp((topKLogits[i] - maxLogit) / t);
        }

        var r = nextFloat() * total;
        var acc = 0.0;
        for (i in 0...sz) {
            acc += Math.exp((topKLogits[i] - maxLogit) / t);
            if (r <= acc) {
                var id = topKIds[i];
                pushRecent(id);
                return id;
            }
        }
        var id = topKIds[sz - 1];
        pushRecent(id);
        return id;
    }

    /**
     * Per-element fallback scan — invoked only when the runtime's bulk
     * `topkScan` reports `-1` (logits not F32-contiguous). Mirrors the
     * original Haxe loop exactly so tie-breaking + top-K output stays
     * byte-identical to the fast path.
     */
    private function topKScanFallback(
        logits:Tensor, n:Int, k:Int, penalize:Bool, rp:Float
    ):Int {
        var sz = 0;
        for (i in 0...n) {
            var lg = adjusted(logits.getFlat(i), i, penalize, rp);
            if (sz < k) {
                var pos = sz;
                while (pos > 0 && topKLogits[pos - 1] < lg) {
                    topKLogits[pos] = topKLogits[pos - 1];
                    topKIds[pos] = topKIds[pos - 1];
                    pos--;
                }
                topKLogits[pos] = lg;
                topKIds[pos] = i;
                sz++;
            } else if (lg > topKLogits[k - 1]) {
                var pos = k - 1;
                while (pos > 0 && topKLogits[pos - 1] < lg) {
                    topKLogits[pos] = topKLogits[pos - 1];
                    topKIds[pos] = topKIds[pos - 1];
                    pos--;
                }
                topKLogits[pos] = lg;
                topKIds[pos] = i;
            }
        }
        return sz;
    }

    /** Apply repetition penalty if `id` is in the recent window. */
    private inline function adjusted(lg:Float, id:Int, penalize:Bool, rp:Float):Float {
        if (!penalize) return lg;
        if (!isRecent(id)) return lg;
        return (lg > 0.0) ? lg / rp : lg * rp;
    }

    /** Linear scan of the 64-entry recent buffer — small enough that
        a hashmap probe would cost more than the loop. */
    private function isRecent(id:Int):Bool {
        for (k in 0...RECENT_CAP) {
            if (recent[k] == id) return true;
        }
        return false;
    }

    /** Rolling write into the recent-id ring buffer. */
    private function pushRecent(id:Int):Void {
        recent[recentWrite] = id;
        recentWrite = (recentWrite + 1) % RECENT_CAP;
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
        //   temp = 0.0 → effective-greedy via LocalTempSampler(epsilon=1e-4)
        //     with NO repetition penalty. Routes around the trap-stub
        //     cascade that ArgmaxSampler currently trips on import (see
        //     bugs_trap_stub_cascade). Numerically equivalent to argmax
        //     for the dominant case (top-1 probability gap >> epsilon).
        //   temp > 0.0 → TemperatureSampler with topK=50 + rep-penalty=1.15
        //     standard chat-quality recipe. O(n) per token.
        //
        // When ArgmaxSampler's import cascade is fixed, restore the
        // direct dispatch: `(temp > 0) ? LocalTempSampler(...) : ArgmaxSampler()`.
        // Until then the LocalTempSampler(1e-4, 1.0, 1, 42) form is
        // deterministic-greedy without tripping the cascade.
        var sampler:Sampler = (temperature > 0.0)
            ? new LocalTempSampler(temperature, 1.15, 50, 42)
            : new LocalTempSampler(1e-4, 1.0, 1, 42);

        // Instruct models behave best when the prompt is wrapped in the
        // model's chat template. Llama-3 uses
        //   <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
        //   {prompt}<|eot_id|>
        //   <|start_header_id|>assistant<|end_header_id|>\n\n
        // The BPE tokenizer recognises registered specials as atomic
        // ids; GGUFTokenizer registers the Llama-3 chat specials by
        // direct vocab lookup so this string round-trips correctly.
        //
        // A system prompt would normally help the model stay in
        // character, but on the 1B Instruct the extra context seems
        // to make the precision drift through 16 layers worse (a
        // greedy-decode run with a short system prompt SIGSEGVs;
        // sampled runs free-associate about whatever the system
        // prompt mentioned). Skip it until the precision-drift work
        // lands.
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

        // Dump the prompt-encoding for diff harnesses
        // (see tools/llama-diff/compare.sh). One token id per line,
        // prefixed with `[dbg.prompt-id]` so the harness can grep it
        // back out without confusing other trace output. Same shape
        // as `[129 ids]` then `12345`, one per line.
        var promptIds = tok.encode(modelPrompt);
        trace("[dbg.prompt-len] " + promptIds.length);
        for (i in 0...promptIds.length) {
            trace("[dbg.prompt-id] " + i + " " + promptIds[i]);
        }

        var loop = new GenerationLoop(model, tok, sampler, eos, maxNew);

        trace("[gen] streaming...");
        // Live print: emit the partial-text delta each step so the
        // user sees tokens appear in real time. The callback's
        // `partial` is the full decoded tail so far, not just the
        // newest piece, so we track the previous length and slice.
        //
        // Counters live in single-element arrays because Rayzor
        // closures capture primitives by value — `prevLen++` inside
        // the closure wouldn't be visible to the outer scope (every
        // tick would see `prevLen = 0` and print the full partial,
        // producing the "TheThe corrected" duplicated-prefix effect).
        // Arrays are heap objects and capture by reference.
        var prevLen = [0];
        var nTokens = [0];
        var startedAt = Sys.time();
        var output = loop.generate(modelPrompt, function(_id:Int, partial:String):Bool {
            var delta = partial.substr(prevLen[0]);
            prevLen[0] = partial.length;
            Sys.print(delta);
            nTokens[0] = nTokens[0] + 1;
            return true;
        });
        var elapsed = Sys.time() - startedAt;
        Sys.println("");
        trace("");
        trace("[done] " + nTokens[0] + " tokens in " + fmt(elapsed) + "s ("
            + fmt(nTokens[0] / elapsed) + " tok/s)");
        trace("[output] " + output);
    }

    static inline function fmt(x:Float):String {
        return Std.string(Math.round(x * 1000) / 1000);
    }
}
