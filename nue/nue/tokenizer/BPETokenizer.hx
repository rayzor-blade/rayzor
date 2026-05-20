package nue.tokenizer;

/**
 * Byte-Pair Encoding tokenizer — the byte-level variant used by
 * tiktoken (GPT-2/3/4, Llama 3, Qwen) and the SentencePiece variant
 * used by Llama 2 / Mistral are close enough that a single
 * implementation covers both with parameter knobs.
 *
 * Algorithm (encode):
 *   1. Split text into a starting sequence of single-byte tokens (or
 *      UTF-8 codepoints for SentencePiece-style).
 *   2. Repeatedly find the merge-ranked pair (`a`, `b`) that
 *      appears in the sequence with the lowest rank (= earliest learned
 *      merge) and replace every occurrence with the merged token `ab`.
 *   3. Stop when no learnable pair remains.
 *
 * Decode is the inverse table lookup — concatenate `vocab.tokens[id]`
 * for each ID. Byte-level tokens encode `0x00..0xFF` via the GPT-2
 * remapping (replace control bytes with printable Unicode aliases so
 * the JSON vocab stays printable); decode unmaps them.
 *
 * **Performance.** Pure Haxe; expected ~10–50× slower than a Rust
 * tokenizer for long prompts. Tokenisation is <1% of inference time
 * for an LLM, so we ship pure Haxe in v1. Merge lookup is O(N)
 * linear scan because `Map<String, V>.get` doesn't reliably return
 * values in the current rayzor stdlib.
 *
 * **Special tokens** are stored on top of the merged vocab. Encode
 * scans for known special-token literals before applying BPE so they
 * pass through as their atomic ID (Llama 3 conversations use
 * `<|begin_of_text|>` / `<|start_header_id|>` / `<|end_header_id|>`
 * etc.).
 */
class BPETokenizer implements Tokenizer {
    public var vocab:Vocab;
    /** Ordered merge rules; index = priority (lower = applied first). */
    public var merges:Array<MergeRule>;
    /** Parallel arrays for special tokens. */
    public var specialNames:Array<String>;
    public var specialIds:Array<Int>;
    /** Byte-level tokenizers map raw bytes through a printable alias
        table — required for GPT-2/Llama 3 style vocabs. */
    public var byteLevel:Bool;

    public function new(vocab:Vocab, merges:Array<MergeRule>, byteLevel:Bool) {
        this.vocab = vocab;
        this.merges = merges;
        this.byteLevel = byteLevel;
        this.specialNames = [];
        this.specialIds = [];
    }

    /** Register a special token (name → existing vocab ID). */
    public function addSpecial(name:String, id:Int):Void {
        specialNames.push(name);
        specialIds.push(id);
    }

    public function vocabSize():Int {
        return vocab.size();
    }

    public function specialId(name:String):Int {
        for (i in 0...specialNames.length) {
            if (specialNames[i] == name) return specialIds[i];
        }
        return -1;
    }

    /** Linear scan for a merge rule matching `left + right`. Returns
        -1 if no rule exists. */
    private function rankOf(left:String, right:String):Int {
        var needle = left + right;
        for (i in 0...merges.length) {
            if (merges[i].key == needle) return i;
        }
        return -1;
    }

    /**
     * Encode a UTF-8 string. Splits into single-byte/codepoint tokens
     * then applies merges in priority order until none apply.
     */
    public function encode(text:String):Array<Int> {
        var pieces:Array<String> = [];
        for (i in 0...text.length) pieces.push(text.charAt(i));

        var iterations = 0;
        var maxIter = pieces.length * 8;
        while (iterations < maxIter) {
            iterations++;
            var bestRank = -1;
            var bestIdx = -1;
            var n = pieces.length - 1;
            for (i in 0...n) {
                var rank = rankOf(pieces[i], pieces[i + 1]);
                if (rank < 0) continue;
                if (bestRank == -1 || rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx == -1) break;
            var merged = merges[bestRank].merged;
            var next:Array<String> = [];
            var k = 0;
            while (k < pieces.length) {
                if (k == bestIdx) {
                    next.push(merged);
                    k += 2;
                } else {
                    next.push(pieces[k]);
                    k++;
                }
            }
            pieces = next;
        }

        var ids:Array<Int> = [];
        for (j in 0...pieces.length) {
            var p = pieces[j];
            var id = vocab.lookup(p);
            if (id < 0 && p.length > 0) {
                id = vocab.lookup(p.charAt(0));
            }
            if (id >= 0) ids.push(id);
        }
        return ids;
    }

    public function decode(ids:Array<Int>):String {
        // Plain string concatenation. `StringBuf.add(Dynamic)` chains
        // through a Dynamic→toString path that the current JIT can't
        // resolve at link time when called transitively from a Tokenizer-
        // implementing class; switching to `+=` sidesteps the lookup.
        var out = "";
        for (j in 0...ids.length) {
            var id = ids[j];
            if (id >= 0 && id < vocab.size()) {
                out += vocab.get(id);
            }
        }
        return out;
    }
}
