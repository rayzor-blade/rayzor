import nue.loader.GGUFLoader;
import nue.tokenizer.BPETokenizer;
import nue.chat.Conversation;
import nue.arch.LlamaModel;
import rayzor.concurrent.Arc;
import rayzor.concurrent.Channel;
import rayzor.concurrent.Thread;

@:derive([Send])
class StreamMsg {
    public var text:String;
    public var done:Bool;
    public function new(text:String, done:Bool) {
        this.text = text;
        this.done = done;
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
        // Repetition penalty (arg[5]). Divides positive logits / multiplies
        // negative logits for any token in the recent window before Top-K, so
        // a recently-emitted token is less likely to be re-picked. 1.0 disables
        // it; 1.1–1.3 is the typical chat range, higher = more aggressive
        // de-looping. Default 1.3: the 1B model (and the wasm path, whose f32
        // reduction order drifts from native and lands on a more loop-prone
        // greedy path) needs more than the old 1.15 to break sentence loops.
        // NB: this default MUST match the doc above — a 1.0 here (penalty off)
        // is what lets the 1B model collapse into "In conclusion… In
        // conclusion…" loops on long open-ended generations.
        var repPenalty:Float = 1.3;
        if (args.length > 5) {
            var rp = Std.parseFloat(args[5]);
            if (rp == rp && rp >= 1.0) repPenalty = rp;
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
        // LoadedModel.tokenizer is the format-agnostic Tokenizer interface; the
        // llama generation loops take the concrete BPETokenizer. Recover it the
        // same checked way as the model cast on the next line.
        var tok = cast(loaded.tokenizer, BPETokenizer);
        var llama = cast(loaded.model, LlamaModel);
        var profilePool = truthyEnv("RAYZOR_PROFILE_POOL");

        trace("[meta] " + meta.architecture + " hidden=" + meta.hiddenSize
            + " layers=" + meta.numLayers + " heads=" + meta.numHeads
            + "/" + meta.numKvHeads + " ffn=" + meta.intermediateSize
            + " vocab=" + meta.vocabSize + " ctx=" + meta.maxSeqLen
            + " headDim=" + meta.headDim + " ropeBase=" + meta.ropeBase
            + " normEps=" + meta.normEps);
        if (llama.spinPool != null) {
            trace("[pool] workers=" + llama.spinPool.workers());
        }
        trace("[tok]  vocab=" + tok.vocabSize());

        // Build the conversation. The arch-aware chat template, per-arch
        // stop tokens, and sampling config (rep-penalty 1.3, which breaks the
        // 1B "In conclusion…" loops) all live in nue.chat.Conversation now.
        var chat = Conversation.fromLoaded(llama, tok, meta);
        chat.config.maxNewTokens = maxNew;
        chat.config.temperature = temperature;
        chat.config.repetitionPenalty = repPenalty;

        var stops = chat.stopIds();
        trace("[tok]  eos=" + (stops.length > 0 ? stops[0] : -1));
        trace("[chat] template=" + chat.template.kind);
        trace("");

        trace("rep-penalty: " + repPenalty + "  + no-repeat-8gram");

        // Prompt-encoding dump for the tokenizer diff harness (see
        // tools/llama-diff/compare.sh). Gated behind RAYZOR_LLAMA_DUMP_PROMPT.
        if (Sys.getEnvOr("NUE_LLAMA_DUMP_PROMPT", "RAYZOR_LLAMA_DUMP_PROMPT") != null) {
            var rendered = chat.nextPrompt(prompt);
            var promptIds = tok.encode(rendered);
            trace("[dbg.prompt-len] " + promptIds.length);
            for (i in 0...promptIds.length) {
                trace("[dbg.prompt-id] " + i + " " + promptIds[i]);
            }
        }

        var specEnv = Sys.getEnvOr("NUE_SPEC_DECODE", "RAYZOR_SPEC_DECODE");
        var specOn = specEnv != null && specEnv != "0" && specEnv != ""
            && specEnv.toLowerCase() != "false";
        chat.config.useSpeculative = specOn;
        var silentStream = truthyEnv("RAYZOR_LLAMA_SILENT_STREAM");
        var streamFlushMs = envInt("RAYZOR_STDOUT_FLUSH_MS", 50);

        trace("[gen] streaming...");
        // Live print: the callback receives each token's DELTA directly
        // (GenerationLoop streams deltas, not the cumulative text), so we
        // just print it as tokens appear.
        //
        // The counter lives in a single-element array because Rayzor closures
        // capture primitives by value — `nTokens++` inside the closure wouldn't
        // be visible to the outer scope. Arrays capture by reference.
        var nTokens = [0];
        var firstTokenAt = [0.0];
        var startedAt = Sys.time();
        var streamBuf:Array<Array<String>> = [[]];
        var lastStreamFlush = [startedAt];
        var writerOff = Sys.getEnvOr("NUE_STREAM_WRITER", "RAYZOR_STREAM_WRITER") == "0";
        var streamArc:Arc<Channel<StreamMsg>> = null;
        var writer:Thread<Int> = null;
        if (!silentStream && !writerOff) {
            // Unbounded channel (capacity 0): `send` never blocks, so the
            // decode thread is fully decoupled from terminal I/O even when the
            // tty applies flow control. A bounded channel would block `send`
            // once full, stalling decode at the terminal's render rate.
            var chArc = new Arc(new Channel(0));
            var writerCh = chArc.clone();
            streamArc = chArc;
            writer = Thread.spawn(function():Int {
                // Coalesce: block for one message, then drain everything queued
                // into a single print. Fewer, larger writes cut per-write tty
                // overhead so the writer keeps pace with a fast producer.
                while (true) {
                    var m:StreamMsg = writerCh.get().receive();
                    if (m == null || m.done) break;
                    var out = m.text;
                    var n:StreamMsg = writerCh.get().tryReceive();
                    while (n != null) {
                        if (n.done) { Sys.print(out); return 0; }
                        out += n.text;
                        n = writerCh.get().tryReceive();
                    }
                    Sys.print(out);
                }
                return 0;
            });
        }
        var flushStream = function(force:Bool):Void {
            if (silentStream || streamBuf[0].length == 0) return;
            var now = Sys.time();
            if (!force && streamFlushMs > 0 && (now - lastStreamFlush[0]) * 1000.0 < streamFlushMs) {
                return;
            }
            var chunk = streamBuf[0].join("");
            streamBuf[0] = [];
            lastStreamFlush[0] = now;
            if (streamArc != null) streamArc.get().send(new StreamMsg(chunk, false));
            else Sys.print(chunk);
        }
        var emitToken = function(_id:Int, delta:String):Bool {
            if (firstTokenAt[0] == 0.0) firstTokenAt[0] = Sys.time();
            if (!silentStream) {
                streamBuf[0].push(delta);
                flushStream(streamFlushMs <= 0 || delta.indexOf("\n") >= 0);
            }
            nTokens[0] = nTokens[0] + 1;
            return true;
        };
        var response = chat.ask(prompt, emitToken);
        var output:String = response.text;
        flushStream(true);
        if (streamArc != null) {
            streamArc.get().send(new StreamMsg("", true));
            writer.join();
        }
        var elapsed = Sys.time() - startedAt;
        if (!silentStream) Sys.println("");
        trace("");
        var ttft = (firstTokenAt[0] > 0.0) ? (firstTokenAt[0] - startedAt) : elapsed;
        trace("[done] " + nTokens[0] + " tokens in " + fmt(elapsed) + "s ("
            + fmt(nTokens[0] / elapsed) + " tok/s, ttft=" + fmt(ttft) + "s, finish="
            + response.finishReasonName() + ")");
        if (profilePool && llama.spinPool != null) {
            trace("[profile-pool] " + llama.spinPool.profReport());
        }
        // Which kernel actually ran, per quant scheme (NUE_DUMP_Q4_GATES=1).
        // A pure-Haxe claim is only credible with ffi=0 printed here.
        nue.Q4Matmul.dumpCensus();
        // The streaming callback above already printed every token. Gate
        // the full-text dump behind RAYZOR_LLAMA_DUMP_OUTPUT=1 for
        // diagnostic runs (e.g. when comparing decoded text against
        // llama.cpp byte-for-byte). Default off — users see the stream
        // exactly once.
        if (Sys.getEnvOr("NUE_LLAMA_DUMP_OUTPUT", "RAYZOR_LLAMA_DUMP_OUTPUT") != null) {
            trace("[output] " + output);
        }
        // Join the Haxe-matmul spin pool's workers (no-op on the FFI path):
        // the runtime waits on all live threads before JIT teardown.
        llama.shutdownPool();
    }

    static inline function fmt(x:Float):String {
        return Std.string(Math.round(x * 1000) / 1000);
    }

    static function truthyEnv(name:String):Bool {
        var v = Sys.getEnv(name);
        return v != null && v != "0" && v != "" && v.toLowerCase() != "false";
    }

    static function envInt(name:String, fallback:Int):Int {
        var v = Sys.getEnv(name);
        if (v == null || v == "") return fallback;
        var parsed = Std.parseInt(v);
        if (parsed == null) return fallback;
        return parsed + 0;
    }

}
