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

    /** Clear every layer's KV cache. Call between independent prompts. */
    function resetCache():Void;
}
