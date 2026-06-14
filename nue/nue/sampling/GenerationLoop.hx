package nue.sampling;

import nue.CausalLanguageModel;
import nue.tokenizer.Tokenizer;
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
    public var tokenizer:Tokenizer;
    public var sampler:Sampler;
    public var eosId:Int;
    public var maxNewTokens:Int;

    public function new(
        model:CausalLanguageModel,
        tokenizer:Tokenizer,
        sampler:Sampler,
        eosId:Int,
        maxNewTokens:Int
    ) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.sampler = sampler;
        this.eosId = eosId;
        this.maxNewTokens = maxNewTokens;
    }

    /**
     * Generate text starting from `prompt`. Returns the full string
     * (prompt + generated tail). `onToken` is invoked once per new
     * token with the generated ID and the cumulative text so far —
     * returning `false` from it aborts before the EOS / token limit.
     */
    public function generate(prompt:String, onToken:Int->String->Bool):String {
        // Per-phase decode timing, env-gated. When `RAYZOR_PROFILE_DECODE`
        // is set we accumulate wall time on each Haxe-source-level phase
        // (forward, lastRow, sample, decode_str, free_logits) and emit a
        // single `[profile-decode]` line at exit. When enabled, also
        // records per-step wall time so long runs can report tail latency
        // (p50/p95/p99/max) instead of only final tok/s.
        var _profileOn = Sys.getEnv("RAYZOR_PROFILE_DECODE") != null;
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
        if (_profileOn) {
            _pStepHist = [];
            for (_ in 0..._pHistBuckets) _pStepHist.push(0);
        }

        model.resetCache();

        var _tPrefill = _profileOn ? Sys.time() : 0.0;
        var ids = tokenizer.encode(prompt);
        if (_profileOn) _pTokenize = Sys.time() - _tPrefill;
        if (ids.length == 0) return prompt;
        _pPromptTokens = ids.length;

        // Prefill: feed the entire prompt; take the last row of logits
        // as the prediction for "what comes after the prompt".
        _tPrefill = _profileOn ? Sys.time() : 0.0;
        var logits = model.forwardIds(ids);
        if (_profileOn) _pPrefillForward = Sys.time() - _tPrefill;
        _tPrefill = _profileOn ? Sys.time() : 0.0;
        var lr0 = lastRow(logits);
        if (_profileOn) _pPrefillLastRow = Sys.time() - _tPrefill;
        _tPrefill = _profileOn ? Sys.time() : 0.0;
        var nextId = sampler.sample(lr0);
        if (_profileOn) _pPrefillSample = Sys.time() - _tPrefill;
        if (lr0 != logits) lr0.free();

        // Accumulate the decoded byte-alphabet pieces incrementally. Re-
        // decoding the whole id list per token (`tokenizer.decode(generated)`)
        // is O(N²) per call → O(N³) over a generation, and that allocation
        // churn is the dominant pressure on long streams. Appending one piece
        // per token and `decodeBuffer`-ing the accumulation keeps the output
        // byte-identical (byteDecode still runs on the full buffer, so multi-
        // byte codepoints that straddle a token boundary decode correctly).
        var rawPieces = "";
        var step = 0;
        var _decodeStart = _profileOn ? Sys.time() : 0.0;
        while (true) {
            if (nextId == eosId) break;
            if (maxNewTokens > 0 && step >= maxNewTokens) break;

            var _stepStart = _profileOn ? Sys.time() : 0.0;
            var _stepIndex = _pCount;

            rawPieces += tokenizer.decodePiece(nextId);
            ids.push(nextId);

            // Stream the token through the callback. `partial` is the full
            // decoded text so far, built from the incrementally-accumulated
            // byte-alphabet pieces (no per-token re-decode of the whole list).
            if (onToken != null) {
                var _t0 = _profileOn ? Sys.time() : 0.0;
                var partial = tokenizer.decodeBuffer(rawPieces);
                if (_profileOn) _pDecodeStr += Sys.time() - _t0;
                if (!onToken(nextId, partial)) break;
            }

            step++;
            _pCount++;

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
            if (_profileOn) _pFreeLogits += Sys.time() - _t1;

            _t1 = _profileOn ? Sys.time() : 0.0;
            logits = model.forwardIds([nextId]);
            if (_profileOn) _pForward += Sys.time() - _t1;

            _t1 = _profileOn ? Sys.time() : 0.0;
            var lrN = lastRow(logits);
            if (_profileOn) _pLastRow += Sys.time() - _t1;

            _t1 = _profileOn ? Sys.time() : 0.0;
            nextId = sampler.sample(lrN);
            if (_profileOn) _pSample += Sys.time() - _t1;

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
            }
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
        return prompt + tokenizer.decodeBuffer(rawPieces);
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

}
