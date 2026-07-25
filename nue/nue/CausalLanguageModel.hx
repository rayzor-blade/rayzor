package nue;

import rayzor.ds.Tensor;

/**
 * `Module` specialised for **causal** language models — the autoregressive
 * decoder shape used by GPT, Llama, Mistral, Qwen, Phi, Falcon, …
 *
 * Adds two conveniences every causal generation loop needs:
 *   - `forwardIds`: take an `Array<Int>` of token IDs (skips the
 *     intermediate Tensor wrap of `Module.forward`).
 *   - `resetCache`: clear all per-layer KV caches between independent
 *     prompts so the new prefill doesn't see prior context.
 *
 * Non-causal model shapes have their own interfaces:
 *   - `EncoderModel` — bidirectional encoder (BERT, RoBERTa, DeBERTa)
 *   - `Seq2SeqModel` — encoder/decoder pair (T5, BART)  *(future)*
 *
 * Concrete implementations live under `nue.arch.*` (one per
 * architecture family). Loaders return the relevant interface so user
 * code doesn't need to know which family it got back beyond its shape.
 */
interface CausalLanguageModel extends Module {
    /**
     * Forward pass on a sequence of token IDs.
     * Returns logits of shape `[seq_len, vocab_size]`.
     */
    function forwardIds(tokenIds:Array<Int>):Tensor;

    /**
     * Forward pass on a sequence of token IDs, returning only the logits for
     * the final position as `[1, vocab_size]`. The full sequence still flows
     * through the transformer blocks so KV cache prefill remains correct.
     */
    function forwardLastLogits(tokenIds:Array<Int>):Tensor;

    /** Clear every layer's KV cache. Call between independent prompts. */
    function resetCache():Void;

    /** Number of tokens currently held in the KV cache. */
    function cacheLen():Int;

    /**
     * Maximum tokens the KV cache can hold — the usable context window.
     * Generation loops stop before exceeding this so an append never
     * overflows the pre-sized cache.
     */
    function cacheCapacity():Int;

    /**
     * Drop KV entries back to `len` tokens (roll the cache back to an
     * earlier position). Used by speculative decoding to discard rejected
     * draft tokens.
     */
    function rewindCache(len:Int):Void;
}
