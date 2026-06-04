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
        model.resetCache();

        var ids = tokenizer.encode(prompt);
        if (ids.length == 0) return prompt;

        // Prefill: feed the entire prompt; take the last row of logits
        // as the prediction for "what comes after the prompt".
        var logits = model.forwardIds(ids);
        var nextId = sampler.sample(lastRow(logits));

        var generated:Array<Int> = [];
        var step = 0;
        while (true) {
            if (nextId == eosId) break;
            if (maxNewTokens > 0 && step >= maxNewTokens) break;

            generated.push(nextId);
            ids.push(nextId);

            // Stream the token through the callback. Build the
            // partial text by decoding the freshly generated tail
            // only — decoding the entire ids array per token would
            // be O(N²) over generation length.
            if (onToken != null) {
                var partial = tokenizer.decode(generated);
                if (!onToken(nextId, partial)) break;
            }

            step++;

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
            logits = model.forwardIds([nextId]);
            nextId = sampler.sample(lastRow(logits));
        }

        return prompt + tokenizer.decode(generated);
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
        // to `[vocab_size]` before handing off.
        var vocab = shape[shape.length - 1];
        return logits.slice(0, lastIdx, lastIdx + 1).reshape([vocab]);
    }
}
