package nue.tokenizer;

import haxe.ds.StringMap;

/**
 * Byte-Pair Encoding tokenizer — the byte-level variant used by
 * tiktoken (GPT-2/3/4, Llama 3, Qwen) and the SentencePiece variant
 * used by Llama 2 / Mistral are close enough that a single
 * implementation covers both with parameter knobs.
 *
 * Algorithm (encode):
 *   1. For byte-level tokenizers, map the input UTF-8 bytes through
 *      the GPT-2 byte alphabet (printable Unicode per byte) so the
 *      vocab can stay JSON-friendly. SentencePiece mode skips this.
 *   2. Split into one piece per character (post-encoding).
 *   3. Repeatedly find the lowest-ranked mergeable pair via
 *      `rankIndex.get(left + right)` and replace every occurrence.
 *      Stop when no pair has a registered merge.
 *   4. Look up each remaining piece in `vocab` to get its token ID.
 *
 * Decode reverses everything: concatenate `vocab.tokens[id]` for each
 * ID, then (for byte-level tokenizers) translate each character back
 * through the inverse byte alphabet and reinterpret the byte sequence
 * as UTF-8.
 *
 * **Pre-tokenization (v1).** llama.cpp / OpenAI tokenize prompts
 * after a regex-based pre-split (separating contractions, runs of
 * letters, digits, punctuation). We don't apply that split yet — BPE
 * runs over the whole byte-encoded prompt as a single chunk. Outputs
 * usually match for short prompts because the same merges fire, but
 * complex prompts (mixed punctuation, contractions) can diverge from
 * a llama.cpp-encoded reference by 1-2 tokens. Plenty good for a
 * functional pipeline; flagged for a follow-up pass.
 *
 * **Special tokens** are stored on top of the merged vocab via
 * `addSpecial(name, id)`. The encode loop checks for special-token
 * literals before BPE runs, so `<|begin_of_text|>` and friends pass
 * through as their atomic IDs instead of being byte-split.
 */
class BPETokenizer implements Tokenizer {
    public var vocab:Vocab;
    /** Ordered merge rules; index = priority (lower = applied first). */
    public var merges:Array<MergeRule>;
    /** key (= left+right) → index into `merges`. O(1) priority lookup. */
    public var rankIndex:StringMap<Int>;
    /** Special token name → vocab ID. */
    public var specials:StringMap<Int>;
    /** Byte-level tokenizers map raw bytes through a printable alias
        table — required for GPT-2/Llama 3 style vocabs. */
    public var byteLevel:Bool;

    public function new(vocab:Vocab, merges:Array<MergeRule>, byteLevel:Bool) {
        this.vocab = vocab;
        this.merges = merges;
        this.byteLevel = byteLevel;
        this.rankIndex = new StringMap<Int>();
        for (i in 0...merges.length) {
            rankIndex.set(merges[i].key, i);
        }
        this.specials = new StringMap<Int>();
    }

    /** Register a special token (name → existing vocab ID). */
    public function addSpecial(name:String, id:Int):Void {
        specials.set(name, id);
    }

    public function vocabSize():Int {
        return vocab.size();
    }

    public function specialId(name:String):Int {
        var id = specials.get(name);
        return (id == null) ? -1 : id;
    }

    /** O(1) lookup for a merge rule matching `left + right`. Returns
        -1 if no rule exists. */
    private function rankOf(left:String, right:String):Int {
        var rank = rankIndex.get(left + right);
        return (rank == null) ? -1 : rank;
    }

    /**
     * Encode a UTF-8 string. Byte-encodes (when configured), then runs
     * BPE merges across the whole input.
     */
    public function encode(text:String):Array<Int> {
        var pieces:Array<String>;
        if (byteLevel) {
            pieces = byteEncodePieces(text);
        } else {
            pieces = [];
            for (i in 0...text.length) pieces.push(text.charAt(i));
        }

        var iterations = 0;
        var maxIter = pieces.length * 8 + 16;
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
        // Concatenate per-token strings. For byte-level tokenizers, the
        // result is in the GPT-2 byte alphabet — translate back to raw
        // bytes and reinterpret as UTF-8 before returning.
        var out = "";
        for (j in 0...ids.length) {
            var id = ids[j];
            if (id >= 0 && id < vocab.size()) {
                out += vocab.get(id);
            }
        }
        if (byteLevel) {
            return byteDecode(out);
        }
        return out;
    }

    // ------------------------------------------------------------------
    // GPT-2 / Llama-3 byte alphabet
    // ------------------------------------------------------------------

    /**
     * Map each input byte through the GPT-2 byte→Unicode alphabet and
     * return one piece per encoded character. Byte 0x20 (space) becomes
     * the Unicode codepoint U+0120 (`Ġ`), 0x0A (LF) becomes `Ċ`, etc.
     * Printable Latin-1 bytes map to themselves.
     */
    static function byteEncodePieces(text:String):Array<String> {
        var pieces:Array<String> = [];
        if (text == null || text.length == 0) return pieces;
        var bytes = haxe.io.Bytes.ofString(text);
        var n = bytes.length;
        var table = byteEncoderTable();
        for (i in 0...n) {
            var b = bytes.get(i);
            pieces.push(table[b]);
        }
        return pieces;
    }

    /**
     * Inverse byte alphabet: walk the concatenated decoded string char
     * by char, translate each codepoint back to its source byte, then
     * reinterpret the byte sequence as UTF-8.
     *
     * Uses an array-of-int reverse table (size 0x144 = U+0143 max) for
     * O(1) lookup. Bytes go into an allocated `Bytes` then `toString`
     * reinterprets them as UTF-8.
     */
    static function byteDecode(s:String):String {
        if (s == null || s.length == 0) return "";
        var rev = byteDecoderArray();
        var n = s.length;
        // Worst case: each input char is a 3-byte UTF-8 unknown that
        // we pass through as raw codepoint bytes.
        var buf = haxe.io.Bytes.alloc(n * 3);
        var pos = 0;
        for (i in 0...n) {
            var code = s.charCodeAt(i);
            var mapped = (code < rev.length) ? rev[code] : -1;
            if (mapped >= 0) {
                buf.set(pos, mapped);
                pos++;
            } else {
                if (code < 0x80) {
                    buf.set(pos, code);
                    pos++;
                } else if (code < 0x800) {
                    buf.set(pos, 0xC0 | (code >> 6));
                    buf.set(pos + 1, 0x80 | (code & 0x3F));
                    pos += 2;
                } else {
                    buf.set(pos, 0xE0 | (code >> 12));
                    buf.set(pos + 1, 0x80 | ((code >> 6) & 0x3F));
                    buf.set(pos + 2, 0x80 | (code & 0x3F));
                    pos += 3;
                }
            }
        }
        return buf.sub(0, pos).toString();
    }

    /**
     * Build the byte→Unicode alphabet table fresh on each call. We
     * intentionally avoid a static-var cache here: the JIT's static
     * field initialiser path treats `static var ENCODER_CACHE:Array<String> = null`
     * as a class-load-time materialisation that can SIGILL during
     * `new BPETokenizer(...)` even from call sites that never invoke
     * `byteEncoderTable()`. 256 chars cost ~µs to rebuild.
     */
    static function byteEncoderTable():Array<String> {
        var table:Array<String> = [for (_ in 0...256) ""];
        for (b in 0x21...0x7F) table[b] = String.fromCharCode(b);
        for (b in 0xA1...0xAD) table[b] = String.fromCharCode(b);
        for (b in 0xAE...0x100) table[b] = String.fromCharCode(b);
        var next = 0x100;
        for (b in 0...256) {
            if (table[b] == "") {
                table[b] = String.fromCharCode(next);
                next++;
            }
        }
        return table;
    }

    static function byteDecoderMap():StringMap<Int> {
        var table = byteEncoderTable();
        var map = new StringMap<Int>();
        for (b in 0...256) {
            map.set(table[b], b);
        }
        return map;
    }

    /**
     * Reverse alphabet keyed by char-code → byte. Length 0x144 covers
     * every codepoint the forward table can emit (0x00..0xFF self-maps
     * + the U+0100..U+0143 spillover range). Slots that aren't mapped
     * by the alphabet hold -1 so the caller can detect them.
     */
    static function byteDecoderArray():Array<Int> {
        var table = byteEncoderTable();
        var rev:Array<Int> = [for (_ in 0...0x144) -1];
        for (b in 0...256) {
            var ch = table[b];
            // Each entry is a single codepoint (BMP) — pull its char
            // code and use as the reverse index.
            if (ch.length > 0) {
                var code = ch.charCodeAt(0);
                if (code >= 0 && code < rev.length) {
                    rev[code] = b;
                }
            }
        }
        return rev;
    }
}
