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
    /** Parallel array of registered special-token names. `StringMap`
        doesn't expose stable key iteration in Rayzor, so we mirror the
        names here for the encoder's inline special-detection pass. */
    public var specialNames:Array<String>;
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
        this.specialNames = [];
    }

    /** Register a special token (name → existing vocab ID). */
    public function addSpecial(name:String, id:Int):Void {
        if (!specials.exists(name)) specialNames.push(name);
        specials.set(name, id);
    }

    public function vocabSize():Int {
        return vocab.size();
    }

    public function specialId(name:String):Int {
        // See `rankOf` for why this uses `exists` instead of `Null<T>`.
        if (!specials.exists(name)) return -1;
        return specials.get(name);
    }

    /** O(1) lookup for a merge rule matching `left + right`. Returns
        -1 if no rule exists.
        Uses an explicit `exists` check rather than relying on the
        `Null<Int>` semantics of `StringMap.get` — the runtime returns
        raw `0u64` for missing keys (not boxed null), so `get(...) ==
        null` would be `false` for missing keys and miscompare against
        a real 0 entry. See `bugs_stringmap_null_get.md`. */
    private function rankOf(left:String, right:String):Int {
        var key = left + right;
        if (!rankIndex.exists(key)) return -1;
        return rankIndex.get(key);
    }

    /**
     * Encode a UTF-8 string. Byte-encodes (when configured), then runs
     * BPE merges across the whole input.
     *
     * Special tokens registered via `addSpecial` are recognised as
     * literal substrings — the encoder walks the input, emits each
     * special's atomic ID when matched (longest-prefix wins on
     * collision), and BPE-encodes the runs of normal text between
     * specials. This is what lets chat-template strings like
     * `<|begin_of_text|>...<|eot_id|>` round-trip cleanly without
     * the literal `<|...|>` getting byte-split.
     */
    public function encode(text:String):Array<Int> {
        if (specialNames.length == 0) {
            return encodeBPE(text);
        }

        var ids:Array<Int> = [];
        var i = 0;
        var n = text.length;
        var chunkStart = 0;
        while (i < n) {
            var bestMatched = -1;
            var bestLen = 0;
            for (k in 0...specialNames.length) {
                var name = specialNames[k];
                var slen = name.length;
                if (slen <= bestLen) continue;
                if (i + slen > n) continue;
                if (substringEq(text, i, name)) {
                    bestMatched = specials.get(name);
                    bestLen = slen;
                }
            }
            if (bestMatched >= 0) {
                if (i > chunkStart) {
                    var sub = text.substr(chunkStart, i - chunkStart);
                    var subIds = encodeBPE(sub);
                    // Index iteration: `for (id in subIds)` desugars to
                    // `haxe.iterators.ArrayIterator.new` which is trap-
                    // stubbed in the current JIT and would SIGILL.
                    for (kk in 0...subIds.length) ids.push(subIds[kk]);
                }
                ids.push(bestMatched);
                i += bestLen;
                chunkStart = i;
            } else {
                i++;
            }
        }
        if (chunkStart < n) {
            var tail = text.substr(chunkStart, n - chunkStart);
            var tailIds = encodeBPE(tail);
            for (kk in 0...tailIds.length) ids.push(tailIds[kk]);
        }
        return ids;
    }

    /** Raw BPE encode of a span — no special-token recognition. */
    private function encodeBPE(text:String):Array<Int> {
        if (text.length == 0) return [];
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

    /** Byte-level substring equality: `text[pos..pos+needle.length] == needle`. */
    private static function substringEq(text:String, pos:Int, needle:String):Bool {
        var nlen = needle.length;
        if (pos + nlen > text.length) return false;
        for (j in 0...nlen) {
            if (text.charCodeAt(pos + j) != needle.charCodeAt(j)) return false;
        }
        return true;
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
     * Inverse byte alphabet: walk the input as UTF-8, decode codepoints,
     * translate each codepoint back to its source byte via `rev`, then
     * reinterpret the byte sequence as UTF-8.
     *
     * CRITICAL: must walk UTF-8 codepoints, NOT raw bytes. Rayzor's
     * `String.charCodeAt(i)` returns the i-th BYTE (Strings are byte-
     * indexed); a naive per-char walk would split `Ġ` (UTF-8 `c4 a0`)
     * into separate `0xC4` and `0xA0` codes. `rev[0xC4]` self-maps to
     * byte 0xC4, `rev[0xA0]` is -1 → fallback writes `c2 a0`, producing
     * invalid UTF-8 (`c4 c2`) → `Bytes.toString` emits `�`. The fix is
     * to combine multi-byte UTF-8 sequences into a single codepoint
     * before the reverse lookup.
     */
    static function byteDecode(s:String):String {
        if (s == null || s.length == 0) return "";
        var rev = byteDecoderArray();
        var n = s.length;
        // Worst case: each codepoint is a 4-byte UTF-8 unknown that
        // we pass through as raw codepoint bytes.
        var buf = haxe.io.Bytes.alloc(n * 4);
        var pos = 0;
        var i = 0;
        while (i < n) {
            var b0 = s.charCodeAt(i);
            var code:Int;
            var consumed:Int;
            if (b0 < 0x80) {
                code = b0;
                consumed = 1;
            } else if ((b0 & 0xE0) == 0xC0 && i + 1 < n) {
                var b1 = s.charCodeAt(i + 1);
                code = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
                consumed = 2;
            } else if ((b0 & 0xF0) == 0xE0 && i + 2 < n) {
                var b1 = s.charCodeAt(i + 1);
                var b2 = s.charCodeAt(i + 2);
                code = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
                consumed = 3;
            } else if ((b0 & 0xF8) == 0xF0 && i + 3 < n) {
                var b1 = s.charCodeAt(i + 1);
                var b2 = s.charCodeAt(i + 2);
                var b3 = s.charCodeAt(i + 3);
                code = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
                consumed = 4;
            } else {
                // Malformed lead byte — pass through and advance one.
                code = b0;
                consumed = 1;
            }

            var mapped = (code < rev.length) ? rev[code] : -1;
            if (mapped >= 0) {
                buf.set(pos, mapped);
                pos++;
            } else {
                // Unknown codepoint — re-emit as UTF-8 verbatim.
                if (code < 0x80) {
                    buf.set(pos, code);
                    pos++;
                } else if (code < 0x800) {
                    buf.set(pos, 0xC0 | (code >> 6));
                    buf.set(pos + 1, 0x80 | (code & 0x3F));
                    pos += 2;
                } else if (code < 0x10000) {
                    buf.set(pos, 0xE0 | (code >> 12));
                    buf.set(pos + 1, 0x80 | ((code >> 6) & 0x3F));
                    buf.set(pos + 2, 0x80 | (code & 0x3F));
                    pos += 3;
                } else {
                    buf.set(pos, 0xF0 | (code >> 18));
                    buf.set(pos + 1, 0x80 | ((code >> 12) & 0x3F));
                    buf.set(pos + 2, 0x80 | ((code >> 6) & 0x3F));
                    buf.set(pos + 3, 0x80 | (code & 0x3F));
                    pos += 4;
                }
            }
            i += consumed;
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
     * Reverse alphabet keyed by codepoint → byte. Length 0x144 covers
     * every codepoint the forward table can emit (0x00..0xFF self-maps
     * + the U+0100..U+0143 spillover range). Slots that aren't mapped
     * by the alphabet hold -1 so the caller can detect them.
     *
     * CRITICAL: Rayzor strings are UTF-8 byte arrays and `charCodeAt(i)`
     * returns the i-th byte, not the codepoint. `String.fromCharCode(0x120)`
     * returns a 2-byte UTF-8 string (`c4 a0`), so a naive `ch.charCodeAt(0)`
     * would key the reverse map at 0xC4 instead of 0x120 — silently
     * scrambling the table. Decode the codepoint from the UTF-8 bytes
     * explicitly.
     */
    static function byteDecoderArray():Array<Int> {
        var table = byteEncoderTable();
        var rev:Array<Int> = [for (_ in 0...0x144) -1];
        for (b in 0...256) {
            var ch = table[b];
            if (ch.length == 0) continue;
            var b0 = ch.charCodeAt(0);
            var code:Int;
            if (b0 < 0x80) {
                code = b0;
            } else if ((b0 & 0xE0) == 0xC0 && ch.length >= 2) {
                var b1 = ch.charCodeAt(1);
                code = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
            } else if ((b0 & 0xF0) == 0xE0 && ch.length >= 3) {
                var b1 = ch.charCodeAt(1);
                var b2 = ch.charCodeAt(2);
                code = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
            } else if ((b0 & 0xF8) == 0xF0 && ch.length >= 4) {
                var b1 = ch.charCodeAt(1);
                var b2 = ch.charCodeAt(2);
                var b3 = ch.charCodeAt(3);
                code = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
            } else {
                code = b0;
            }
            if (code >= 0 && code < rev.length) {
                rev[code] = b;
            }
        }
        return rev;
    }
}
