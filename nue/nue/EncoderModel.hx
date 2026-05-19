package nue;

import rayzor.ds.Tensor;

/**
 * `Module` specialised for **bidirectional encoder** models — the
 * BERT / RoBERTa / DeBERTa / ELECTRA family.
 *
 * Encoders read the full input at once (no causality), so they don't
 * need a KV cache or per-token decode loop. The forward pass takes
 * the full token sequence in one shot and returns per-token hidden
 * states (and optionally segment / attention masks).
 *
 * Standard downstream tasks plug different "heads" on top of the
 * encoder:
 *   - **Masked LM** — predict masked tokens (the pre-training task).
 *   - **Token classification** — NER, POS tagging.
 *   - **Sequence classification** — sentiment, NLI, retrieval scoring.
 *   - **Sentence embedding** — pool to a single vector (mean / CLS).
 *
 * The encoder itself just returns hidden states; task-specific heads
 * are user code or future `nue.head.*` modules.
 *
 * BERT-family models use absolute positional embeddings (not RoPE),
 * LayerNorm (not RMSNorm), GELU FFN (not SwiGLU), and full
 * bidirectional attention (no causal mask). These primitives live in
 * the same `nue.transformer.*` package as their decoder counterparts.
 */
interface EncoderModel extends Module {
    /**
     * Encode a sequence of token IDs to per-token hidden states.
     * Returns `[seq_len, hidden_size]`.
     *
     * `attentionMask` optionally marks padding positions (1 = attend,
     * 0 = ignore). Pass `null` to attend to every position.
     */
    function encode(tokenIds:Array<Int>, ?attentionMask:Array<Int>):Tensor;
}
