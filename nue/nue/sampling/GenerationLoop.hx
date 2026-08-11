package nue.sampling;

import nue.CausalLanguageModel;
import nue.tokenizer.BPETokenizer;
import rayzor.Mem;
import rayzor.ds.Tensor;

/**
 * Autoregressive text-generation loop. Wires a `CausalLanguageModel`,
 * a `Tokenizer`, and a `Sampler` into a streaming token producer.
 *
 * Lifecycle:
 *   1. Tokenize the prompt to IDs.
 *   2. Prefill — run the model on every prompt token to seed the KV
 *      cache and produce the first "next-token" logits.
 *   3. Decode loop:
 *      - Sample one token from the latest logits.
 *      - If it equals the configured EOS, stop.
 *      - Otherwise: feed it back into the model (single-token forward;
 *        the KV cache provides the prior context), append to the
 *        running ID list, emit via the optional `onToken` callback.
 *   4. Decode the full ID list back into text.
 *
 * The loop owns the KV cache lifecycle — calls `model.resetCache()`
 * once at the start so each generation begins from a clean slate.
 *
 * **Streaming.** Pass `onToken(id, partialText)` to receive each
 * token as it's emitted (useful for TUI streaming). Returning
 * `false` from the callback aborts generation early.
 *
 * **Limits.** `maxNewTokens` caps the run regardless of EOS; setting
 * it to `0` means "generate until EOS or context exhaustion". A
 * future refinement would expose a `stopTokens: Array<Int>` for
 * arbitrary stop strings; for now EOS is the only stop sentinel.
 */
class GenerationLoop {
    public var model:CausalLanguageModel;
    public var tokenizer:BPETokenizer;
    public var sampler:LocalTempSampler;
    public var eosId:Int;
    public var maxNewTokens:Int;
    // Additional stop-token ids beyond `eosId` (e.g. a template's secondary
    // turn-enders). Empty by default; set before `generate` to honour them.
    public var stopIds:Array<Int>;
    // Why the last `generate` stopped: "stop" (a stop token), "length"
    // (maxNewTokens), "context" (KV cache full), "aborted" (onToken returned
    // false), or "" before the first run. Read after `generate` returns.
    public var finishReason:String;

    public function new(
        model:CausalLanguageModel,
        tokenizer:BPETokenizer,
        sampler:LocalTempSampler,
        eosId:Int,
        maxNewTokens:Int
    ) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.sampler = sampler;
        this.eosId = eosId;
        this.maxNewTokens = maxNewTokens;
        this.stopIds = [];
        this.finishReason = "";
    }

    // A token id ends the turn if it is the primary EOS or any configured
    // secondary stop id.
    inline function isStop(id:Int):Bool {
        if (id == eosId) return true;
        for (s in stopIds) if (id == s) return true;
        return false;
    }

    /**
     * Generate text starting from `prompt`. Returns the full string
     * (prompt + generated tail). `onToken` is invoked once per new token
     * with the generated ID and that token's DELTA (the newly decoded text
     * for this token, not the cumulative text) — concatenating the deltas
     * reproduces the tail. Returning `false` aborts before the EOS / limit.
     */
    public function generate(prompt:String, onToken:Int->String->Bool):String {
        // Per-phase decode timing, env-gated. When `RAYZOR_PROFILE_DECODE`
        // is truthy we accumulate wall time on each Haxe-source-level phase
        // (forward, lastRow, sample, decode_str, free_logits) and emit a
        // single `[profile-decode]` line at exit. When enabled, also
        // records per-step wall time so long runs can report tail latency
        // (p50/p95/p99/max) instead of only final tok/s.
        // VALUE-gated, not presence-gated: `=0`/`=false`/empty/unset all mean
        // OFF (the old `!= null` treated `=0` as enabled — on both backends).
        var _pdEnv = Sys.getEnvOr("NUE_PROFILE_DECODE", "RAYZOR_PROFILE_DECODE");
        var _profileOn = _pdEnv != null && _pdEnv != "0" && _pdEnv != ""
            && _pdEnv.toLowerCase() != "false";
        var _pForward = 0.0;
        var _pLastRow = 0.0;
        var _pSample = 0.0;
        var _pDecodeStr = 0.0;
        var _pFreeLogits = 0.0;
        var _pTokenize = 0.0;
        var _pPrefillForward = 0.0;
        var _pPrefillLastRow = 0.0;
        var _pPrefillSample = 0.0;
        var _pPromptTokens = 0;
        var _pCount = 0;
        var _pHistBuckets = 1024;
        var _pHistScale = 4.0;
        var _pHistMs = 0.25;
        var _pStepHist:Array<Int> = null;
        var _pStepMax = 0.0;
        var _pStepMaxIdx = -1;
        var _pStepMaxStart = 0.0;
        var _pwEnv = Sys.getEnvOr("NUE_PROFILE_DECODE_WINDOWS", "RAYZOR_PROFILE_DECODE_WINDOWS");
        var _profileWindows = _profileOn && _pwEnv != null && _pwEnv != "0"
            && _pwEnv != "" && _pwEnv.toLowerCase() != "false";
        var _windowSize:Int = 64;
        var _pwSize = Sys.getEnvOr("NUE_PROFILE_DECODE_WINDOW", "RAYZOR_PROFILE_DECODE_WINDOW");
        if (_profileWindows && _pwSize != null) {
            var _parsedWindow:Null<Int> = Std.parseInt(_pwSize);
            if (_parsedWindow != null) {
                var _unboxedWindow:Int = _parsedWindow + 0;
                if (_unboxedWindow > 0) _windowSize = _unboxedWindow;
            }
        }
        var _wStartTok = 0;
        var _wStartTime = 0.0;
        var _wForward = 0.0;
        var _wSample = 0.0;
        var _wDecodeStr = 0.0;
        var _wFree = 0.0;
        var _wLastRow = 0.0;
        var _wMax = 0.0;
        if (_profileOn) {
            _pStepHist = [];
            for (_ in 0..._pHistBuckets) _pStepHist.push(0);
        }

        model.resetCache();

        var _tPrefill = _profileOn ? Sys.time() : 0.0;
        var ids:Array<Int> = tokenizer.encode(prompt);
        if (_profileOn) _pTokenize = Sys.time() - _tPrefill;
        if (ids.length == 0) return prompt;
        _pPromptTokens = ids.length;

        // Prefill: the last-logits specialization is an important TTFT lever,
        // but keep it behind an opt-in gate until the sliced-hidden-row view
        // path is proven equivalent for every Haxe matmul kernel. A strided
        // row view fed to raw-pointer Q4 kernels can silently corrupt logits.
        _tPrefill = _profileOn ? Sys.time() : 0.0;
        var logits:Tensor = useFastLastLogits()
            ? model.forwardLastLogits(ids)
            : model.forwardIds(ids);
        if (_profileOn) _pPrefillForward = Sys.time() - _tPrefill;
        _tPrefill = _profileOn ? Sys.time() : 0.0;
        var lr0:Tensor = lastRow(logits);
        if (_profileOn) _pPrefillLastRow = Sys.time() - _tPrefill;
        _tPrefill = _profileOn ? Sys.time() : 0.0;
        var nextId:Int = sampler.sample(lr0);
        if (_profileOn) _pPrefillSample = Sys.time() - _tPrefill;
        if (lr0 != logits) lr0.free();

        // Prefill works in batch-sized activations decode never touches again,
        // and they have just been freed. free() returns those pages to the
        // allocator rather than the OS, so ask for them back once here, on the
        // transition into decode — never inside the token loop, since the call
        // walks the allocator's free regions.
        Mem.releaseFreePages(0);

        // Streaming decode: O(|piece|) per token, O(N) over a generation. The
        // earlier approach kept a growing `rawPieces` String and re-
        // `decodeBuffer`'d the WHOLE buffer every token — O(N²) (on wasm
        // `StringBuf` has no amortised append), the dominant per-token heap
        // leak that capped long generations at ~2 GiB / the 2^31 corruption
        // boundary. Instead carry
        // only the bytes of an output codepoint that straddles a token
        // boundary, emit per-token DELTAS, and accumulate decoded parts in an
        // Array (O(1) push, join once for the return value).
        var parts:Array<String> = [];
        var carryHolder = [haxe.io.Bytes.alloc(0)];
        var skipText = skipTokenText();
        var step = 0;
        // Context bound: the KV cache holds at most `_cacheCap` tokens. Each
        // decode forward appends one; stopping before the cache is full is
        // what turns "generate until context exhaustion" (see class doc) into
        // an actual stop condition instead of a `KVCache overflow` throw.
        var _cacheCap:Int = model.cacheCapacity();
        var _decodeStart = _profileOn ? Sys.time() : 0.0;
        if (_profileWindows) _wStartTime = Sys.time();
        while (true) {
            if (isStop(nextId)) { finishReason = "stop"; break; }
            if (maxNewTokens > 0 && step >= maxNewTokens) { finishReason = "length"; break; }

            var _stepStart = _profileOn ? Sys.time() : 0.0;
            var _stepIndex = _pCount;

            var _t0 = _profileOn ? Sys.time() : 0.0;
            var delta:String = "";
            if (!skipText) {
                delta = tokenizer.decodeStreamStep(carryHolder, nextId);
                parts.push(delta);
            }
            if (_profileOn) {
                var _dt = Sys.time() - _t0;
                _pDecodeStr += _dt;
                if (_profileWindows) _wDecodeStr += _dt;
            }

            // Stream the per-token delta through the callback.
            if (onToken != null) {
                if (!onToken(nextId, delta)) { finishReason = "aborted"; break; }
            }

            step++;
            _pCount++;
            if (maxNewTokens > 0 && step >= maxNewTokens) { finishReason = "length"; break; }

            // Context bound. The forward below appends `nextId` to the KV
            // cache; once the cache is full the next append would throw
            // "KVCache overflow: cap+1 > cap". We've already emitted the
            // current token above, so break here to end cleanly at the
            // context boundary instead of crashing one token past it.
            if (_cacheCap > 0 && model.cacheLen() + 1 > _cacheCap) { finishReason = "context"; break; }

            // Decode step: only feed the latest token; KV cache
            // carries the rest of the context.
            // NB: a fresh `[nextId]` array allocation per step is
            // intentional — reusing a single mutable buffer here
            // (`singleIdBuf[0] = nextId; forwardIds(singleIdBuf);`)
            // triggered `KVCache.append failed (krc=-1, vrc=0)` mid
            // generation, suggesting the model retains a reference to
            // the Array's underlying buffer across iterations.
            // Investigate compiler/runtime escape semantics before
            // trying again.
            //
            // Tensor frees: the previous `logits` (vocab=128k × 4 B
            // = 512 KB) and its lastRow view would otherwise pile up
            // — one per generated token, 300+ MB over a 600-token run.
            // InsertFreePass doesn't recognise tensor returns so we
            // release them inline. The lastRow view sometimes IS the
            // input (1-D logits at prefill seq_len=1), guard the free.
            var _t1 = _profileOn ? Sys.time() : 0.0;
            logits.free();
            if (_profileOn) {
                var _dt = Sys.time() - _t1;
                _pFreeLogits += _dt;
                if (_profileWindows) _wFree += _dt;
            }

            _t1 = _profileOn ? Sys.time() : 0.0;
            logits = model.forwardLastLogits([nextId]);
            if (_profileOn) {
                var _dt = Sys.time() - _t1;
                _pForward += _dt;
                if (_profileWindows) _wForward += _dt;
            }

            _t1 = _profileOn ? Sys.time() : 0.0;
            var lrN:Tensor = lastRow(logits);
            if (_profileOn) {
                var _dt = Sys.time() - _t1;
                _pLastRow += _dt;
                if (_profileWindows) _wLastRow += _dt;
            }

            _t1 = _profileOn ? Sys.time() : 0.0;
            nextId = sampler.sample(lrN);
            if (_profileOn) {
                var _dt = Sys.time() - _t1;
                _pSample += _dt;
                if (_profileWindows) _wSample += _dt;
            }

            if (lrN != logits) lrN.free();

            if (_profileOn) {
                var _stepWall = Sys.time() - _stepStart;
                var _bucket = Std.int(Math.floor(_stepWall * 1000.0 * _pHistScale));
                if (_bucket < 0) _bucket = 0;
                if (_bucket >= _pHistBuckets) _bucket = _pHistBuckets - 1;
                _pStepHist[_bucket] = _pStepHist[_bucket] + 1;
                if (_stepWall > _pStepMax) {
                    _pStepMax = _stepWall;
                    _pStepMaxIdx = _stepIndex;
                    _pStepMaxStart = _stepStart;
                }
                if (_profileWindows) {
                    if (_stepWall > _wMax) _wMax = _stepWall;
                    var _wTokens = _pCount - _wStartTok;
                    if (_wTokens >= _windowSize) {
                        var _wWall = Sys.time() - _wStartTime;
                        trace("[profile-window] tokens=" + _wStartTok + ".." + _pCount
                            + " cache_len=" + model.cacheLen()
                            + " wall_s=" + _wWall
                            + " tok_s=" + (_wTokens / _wWall)
                            + " avg_ms=" + ((_wWall * 1000.0) / _wTokens)
                            + " max_ms=" + (_wMax * 1000.0)
                            + " fwd_s=" + _wForward
                            + " sample_s=" + _wSample
                            + " decodeStr_s=" + _wDecodeStr
                            + " lastRow_s=" + _wLastRow
                            + " free_s=" + _wFree);
                        _wStartTok = _pCount;
                        _wStartTime = Sys.time();
                        _wForward = 0.0;
                        _wSample = 0.0;
                        _wDecodeStr = 0.0;
                        _wLastRow = 0.0;
                        _wFree = 0.0;
                        _wMax = 0.0;
                    }
                }
            }
        }

        if (_profileWindows && _pCount > _wStartTok) {
            var _wTokens = _pCount - _wStartTok;
            var _wWall = Sys.time() - _wStartTime;
            trace("[profile-window] tokens=" + _wStartTok + ".." + _pCount
                + " cache_len=" + model.cacheLen()
                + " wall_s=" + _wWall
                + " tok_s=" + (_wTokens / _wWall)
                + " avg_ms=" + ((_wWall * 1000.0) / _wTokens)
                + " max_ms=" + (_wMax * 1000.0)
                + " fwd_s=" + _wForward
                + " sample_s=" + _wSample
                + " decodeStr_s=" + _wDecodeStr
                + " lastRow_s=" + _wLastRow
                + " free_s=" + _wFree);
        }

        if (_profileOn && _pCount > 0) {
            var _decodeEnd = Sys.time();
            var _wall = _pForward + _pLastRow + _pSample + _pDecodeStr + _pFreeLogits;
            var _target50 = Std.int(Math.floor((_pCount - 1) * 0.50)) + 1;
            var _target95 = Std.int(Math.floor((_pCount - 1) * 0.95)) + 1;
            var _target99 = Std.int(Math.floor((_pCount - 1) * 0.99)) + 1;
            var _p50 = 0.0;
            var _p95 = 0.0;
            var _p99 = 0.0;
            var _cum = 0;
            var _histIdx = 0;
            while (_histIdx < _pHistBuckets) {
                _cum += _pStepHist[_histIdx];
                var _ms = _histIdx * _pHistMs;
                if (_p50 == 0.0 && _cum >= _target50) _p50 = _ms;
                if (_p95 == 0.0 && _cum >= _target95) _p95 = _ms;
                if (_p99 == 0.0 && _cum >= _target99) {
                    _p99 = _ms;
                    break;
                }
                _histIdx++;
            }
            trace("[profile-decode] tokens=" + _pCount
                + " wall_s=" + _wall
                + " fwd_s=" + _pForward
                + " lastRow_s=" + _pLastRow
                + " sample_s=" + _pSample
                + " decodeStr_s=" + _pDecodeStr
                + " free_s=" + _pFreeLogits
                + " prompt_tokens=" + _pPromptTokens
                + " tokenize_s=" + _pTokenize
                + " prefill_s=" + (_pPrefillForward + _pPrefillLastRow + _pPrefillSample)
                + " prefill_fwd_s=" + _pPrefillForward
                + " prefill_lastRow_s=" + _pPrefillLastRow
                + " prefill_sample_s=" + _pPrefillSample
                + " decode_start_s=" + _decodeStart
                + " decode_end_s=" + _decodeEnd
                + " step_p50_ms=" + _p50
                + " step_p95_ms=" + _p95
                + " step_p99_ms=" + _p99
                + " step_max_ms=" + (_pStepMax * 1000.0)
                + " step_max_i=" + _pStepMaxIdx
                + " step_max_s=" + _pStepMaxStart
                + " step_samples=" + _pCount);
        }

        logits.free();
        // Full text = prompt + every emitted delta + any incomplete trailing
        // codepoint still in the carry (flushed leniently, matching what a
        // whole-buffer decodeBuffer would have produced at the end).
        if (skipText) return prompt;
        var tail = (carryHolder[0].length > 0) ? carryHolder[0].toString() : "";
        return prompt + parts.join("") + tail;
    }

    /**
     * Convenience wrapper for callers that don't want streaming —
     * generates silently and returns the final string.
     */
    public function generateSilent(prompt:String):String {
        return generate(prompt, function(_id:Int, _txt:String) return true);
    }

    /**
     * Extract the final row of a `[seq_len, vocab_size]` logits
     * tensor as a 1-D `[vocab_size]` view. The sampler operates on
     * 1-D logits regardless of whether the model ran a prefill batch
     * or a single decode step.
     */
    private static function lastRow(logits:Tensor):Tensor {
        var shape = logits.shape();
        if (shape.length <= 1) return logits;
        if (shape[0] == 1) return logits.reshape([shape[shape.length - 1]]);
        var lastIdx = shape[0] - 1;
        // `slice` keeps the sliced axis (`[lastIdx, lastIdx+1)`),
        // returning shape `[1, vocab_size]`. Samplers index logits
        // with a single coordinate (`logits.get([i])`), so collapse
        // to `[vocab_size]` before handing off. Free the intermediate
        // slice view — reshape on a contiguous source returns a fresh
        // view of the same storage with its own refcount, so the
        // slice's refcount is independently dropped here.
        var vocab = shape[shape.length - 1];
        var sliced = logits.slice(0, lastIdx, lastIdx + 1);
        var reshaped = sliced.reshape([vocab]);
        sliced.free();
        return reshaped;
    }

    private static function useFastLastLogits():Bool {
        var v = Sys.getEnvOr("NUE_PREFILL_LAST_LOGITS", "RAYZOR_PREFILL_LAST_LOGITS");
        return v != null && v != "0" && v != "" && v.toLowerCase() != "false";
    }

    private static function skipTokenText():Bool {
        var silent = Sys.getEnvOr("NUE_LLAMA_SILENT_STREAM", "RAYZOR_LLAMA_SILENT_STREAM");
        if (silent == null || silent == "0" || silent == "" || silent.toLowerCase() == "false") {
            return false;
        }
        return Sys.getEnvOr("NUE_LLAMA_DUMP_OUTPUT", "RAYZOR_LLAMA_DUMP_OUTPUT") == null;
    }

}
