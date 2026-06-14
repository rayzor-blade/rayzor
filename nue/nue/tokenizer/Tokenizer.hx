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

    /**
     * The raw piece for a single token id, BEFORE any byte-level → UTF-8
     * translation. For O(N) streaming decode, append each token's piece to
     * a `StringBuf` and finish with `decodeBuffer` — avoids the O(N²)-per-
     * call re-decode of the whole id list.
     */
    function decodePiece(id:Int):String;

    /**
     * Finish a decode from accumulated `decodePiece` output: applies any
     * byte-level → UTF-8 translation. Must run on the full accumulation (a
     * multi-byte codepoint can straddle a token boundary).
     */
    function decodeBuffer(raw:String):String;

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
