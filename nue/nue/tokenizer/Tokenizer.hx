package nue.tokenizer;

/**
 * Format-agnostic tokenizer interface — every concrete tokenizer
 * (BPE, SentencePiece, WordPiece, tiktoken, …) implements this
 * surface so a generation loop can swap implementations without
 * knowing which scheme produced the IDs.
 *
 * IDs are `Int` because every real-world LLM vocab fits in 31 bits
 * (Llama 3: 128k, Qwen: 152k, GPT-4: ~100k). 64-bit IDs would just
 * burn memory in the KV cache and sampling buffers.
 *
 * Special tokens (BOS, EOS, padding) are exposed by name so the
 * generation loop can prepend BOS and terminate on EOS without
 * hard-coding numeric IDs that change per model.
 */
interface Tokenizer {
    /** Encode a UTF-8 string to a sequence of token IDs. */
    function encode(text:String):Array<Int>;

    /** Decode a sequence of token IDs back to a UTF-8 string. */
    function decode(ids:Array<Int>):String;

    /** Number of tokens in the vocabulary. */
    function vocabSize():Int;

    /**
     * Look up a special token's ID by canonical name
     * (`"<|begin_of_text|>"`, `"<|end_of_text|>"`, `"<s>"`, `"</s>"`,
     * `"<|im_start|>"`, etc.). Returns -1 if the tokenizer doesn't
     * have a token by that name.
     */
    function specialId(name:String):Int;
}
