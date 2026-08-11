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
 * **Pre-tokenization (v1).** Byte-level BPE is normally applied after a
 * regex pre-split (separating contractions, runs of letters, digits,
 * punctuation). We don't apply that split yet — BPE runs over the whole
 * byte-encoded prompt as a single chunk. Short prompts come out the same
 * because the identical merges fire; prompts with mixed punctuation or
 * contractions can differ by 1-2 tokens. Fine for a functional pipeline;
 * flagged for a follow-up pass.
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
    /** SentencePiece (unigram) mode — Llama-1/2, Mistral. Set by GGUFTokenizer
        when `tokenizer.ggml.model == "llama"`. Those GGUFs ship `scores` and NO
        `merges`, so the BPE merge loop has nothing to merge and would emit
        near-character-level tokens. */
    public var spm:Bool = false;
    /** Lazily-built byte→Unicode reverse map, cached per instance. Built
        once (vs the per-call rebuild `byteDecode` did) so streaming decode
        is O(1)/token. Instance field — NOT `static var` — to dodge the JIT
        static-initialiser SIGILL that blocks a static cache here (see
        `byteEncoderTable`'s note). */
    var _byteDecoderCache:Array<Int> = null;

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

    /** Min-heap over (rank, leftSlot) in three parallel Int arrays.

        INSTANCE methods with the comparison INLINED, deliberately. Hand-rolled
        because `Array.sort`'s comparator form has an open codegen defect, and
        instance rather than static because a static->static call inside this
        import-compiled module trips the x-module static-resolution cluster
        (the class traps at runtime rather than failing to compile). Instance
        dispatch is what every other method here already uses. */
    private function heapPush(hr:Array<Int>, hi:Array<Int>, hj:Array<Int>,
                              r:Int, i:Int, j:Int):Void {
        hr.push(r);
        hi.push(i);
        hj.push(j);
        var c = hr.length - 1;
        while (c > 0) {
            var par = (c - 1) >> 1;
            var less = (hr[c] != hr[par]) ? (hr[c] < hr[par]) : (hi[c] < hi[par]);
            if (!less) break;
            var tr = hr[par]; hr[par] = hr[c]; hr[c] = tr;
            var ti = hi[par]; hi[par] = hi[c]; hi[c] = ti;
            var tj = hj[par]; hj[par] = hj[c]; hj[c] = tj;
            c = par;
        }
    }

    private function heapPop(hr:Array<Int>, hi:Array<Int>, hj:Array<Int>):Void {
        var last = hr.length - 1;
        hr[0] = hr[last]; hi[0] = hi[last]; hj[0] = hj[last];
        hr.pop(); hi.pop(); hj.pop();
        var sz = hr.length;
        var par = 0;
        while (2 * par + 1 < sz) {
            var l = 2 * par + 1;
            var rr = l + 1;
            var m = par;
            var lessL = (hr[l] != hr[m]) ? (hr[l] < hr[m]) : (hi[l] < hi[m]);
            m = lessL ? l : m;
            if (rr < sz) {
                var lessR = (hr[rr] != hr[m]) ? (hr[rr] < hr[m]) : (hi[rr] < hi[m]);
                m = lessR ? rr : m;
            }
            if (m == par) break;
            var tr = hr[par]; hr[par] = hr[m]; hr[m] = tr;
            var ti = hi[par]; hi[par] = hi[m]; hi[m] = ti;
            var tj = hj[par]; hj[par] = hj[m]; hj[m] = tj;
            par = m;
        }
    }

    /** U+2581 word mark, built from UTF-8 bytes — no `\u` escape appears
        anywhere else in this tree and the lexer does not accept one. */
    private function wordMark():String {
        var b = haxe.io.Bytes.alloc(3);
        b.set(0, 0xE2);
        b.set(1, 0x96);
        b.set(2, 0x81);
        return b.toString();
    }

    /** Sentinel: this pair is not a vocab entry. */
    private function spmNoMerge():Int {
        return 0x7FFFFFFF;
    }

    /** Split into UTF-8 CHARACTERS — a multi-byte codepoint must stay one
        symbol or merges would straddle it. */
    private function utf8Chars(text:String):Array<String> {
        var out:Array<String> = [];
        var i = 0;
        var n = text.length;
        while (i < n) {
            var c = text.charCodeAt(i);
            var len = (c >= 0xF0) ? 4 : ((c >= 0xE0) ? 3 : ((c >= 0xC0) ? 2 : 1));
            if (i + len > n) len = 1;
            out.push(text.substr(i, len));
            i += len;
        }
        return out;
    }

    /** Merge key: negated, scaled vocab score so the shared MIN-heap pops the
        HIGHEST-scoring merge first. Uses `scoreOf`, NOT `vocab.scores[id]` — a
        cross-module field read. */
    private function spmKey(left:String, right:String):Int {
        var id = vocab.lookup(left + right);
        if (id < 0) return spmNoMerge();
        return Std.int(-vocab.scoreOf(id) * 1000.0);
    }

    private function byteToken(b:Int):String {
        var hex = "0123456789ABCDEF";
        return "<0x" + hex.charAt((b >> 4) & 15) + hex.charAt(b & 15) + ">";
    }

    private function hexVal(c:Int):Int {
        if (c >= 48 && c <= 57) return c - 48;
        if (c >= 65 && c <= 70) return c - 55;
        if (c >= 97 && c <= 102) return c - 87;
        return -1;
    }

    /** SPM piece text -> raw bytes: U+2581 becomes a space, `<0xNN>` becomes
        that byte. Byte-fallback tokens can split a codepoint across tokens, so
        callers must still carry an incomplete tail. */
    private function spmDecodeToBytes(sp:String):haxe.io.Bytes {
        var out:Array<Int> = [];
        var i = 0;
        var n = sp.length;
        while (i < n) {
            var c = sp.charCodeAt(i);
            if (c == 60 && i + 5 < n && sp.charCodeAt(i + 1) == 48
                && (sp.charCodeAt(i + 2) | 32) == 120 && sp.charCodeAt(i + 5) == 62) {
                var hi = hexVal(sp.charCodeAt(i + 3));
                var lo = hexVal(sp.charCodeAt(i + 4));
                if (hi >= 0 && lo >= 0) {
                    out.push((hi << 4) | lo);
                    i += 6;
                    continue;
                }
            }
            if (c == 0xE2 && i + 2 < n && sp.charCodeAt(i + 1) == 0x96
                && sp.charCodeAt(i + 2) == 0x81) {
                out.push(32);
                i += 3;
                continue;
            }
            out.push(c);
            i++;
        }
        var b = haxe.io.Bytes.alloc(out.length);
        for (k in 0...out.length) b.set(k, out[k]);
        return b;
    }

    /** SentencePiece encode. Same machinery as `encodeBPE` — heap over candidate
        pairs, in-place linked list — but a candidate is "left+right exists in the
        vocab" and the order is by vocab SCORE, highest first. Ties go leftmost in
        both. Unknown symbols fall back to per-byte `<0xNN>` tokens. */
    private function encodeSPM(text:String):Array<Int> {
        if (text.length == 0) return [];
        // Space-escaped with a leading word mark, built byte-wise.
        var esc:Array<Int> = [0xE2, 0x96, 0x81];
        for (i in 0...text.length) {
            var c = text.charCodeAt(i);
            if (c == 32) {
                esc.push(0xE2);
                esc.push(0x96);
                esc.push(0x81);
            } else {
                esc.push(c);
            }
        }
        var mb = haxe.io.Bytes.alloc(esc.length);
        for (i in 0...esc.length) mb.set(i, esc[i]);
        var pieces = utf8Chars(mb.toString());
        var n = pieces.length;
        if (n > 1) {
            var prevIx:Array<Int> = [];
            var nextIx:Array<Int> = [];
            var alive:Array<Bool> = [];
            for (i in 0...n) {
                prevIx.push(i - 1);
                nextIx.push(i + 1 < n ? i + 1 : -1);
                alive.push(true);
            }
            var hr:Array<Int> = [];
            var hi:Array<Int> = [];
            var hj:Array<Int> = [];
            for (i in 0...(n - 1)) {
                var k0 = spmKey(pieces[i], pieces[i + 1]);
                if (k0 != spmNoMerge()) heapPush(hr, hi, hj, k0, i, i + 1);
            }
            while (hr.length > 0) {
                var k = hr[0];
                var a = hi[0];
                var b = hj[0];
                heapPop(hr, hi, hj);
                if (!alive[a] || !alive[b]) continue;
                if (nextIx[a] != b) continue;
                if (spmKey(pieces[a], pieces[b]) != k) continue;
                pieces[a] = pieces[a] + pieces[b];
                alive[b] = false;
                var bn = nextIx[b];
                nextIx[a] = bn;
                if (bn >= 0) prevIx[bn] = a;
                var ap = prevIx[a];
                if (ap >= 0) {
                    var kl = spmKey(pieces[ap], pieces[a]);
                    if (kl != spmNoMerge()) heapPush(hr, hi, hj, kl, ap, a);
                }
                if (bn >= 0) {
                    var kt = spmKey(pieces[a], pieces[bn]);
                    if (kt != spmNoMerge()) heapPush(hr, hi, hj, kt, a, bn);
                }
            }
            var outp:Array<String> = [];
            var cur = 0;
            while (cur >= 0) {
                outp.push(pieces[cur]);
                cur = nextIx[cur];
            }
            pieces = outp;
        }
        var ids:Array<Int> = [];
        for (j in 0...pieces.length) {
            var p = pieces[j];
            var id = vocab.lookup(p);
            if (id >= 0) {
                ids.push(id);
                continue;
            }
            for (b in 0...p.length) {
                var bid = vocab.lookup(byteToken(p.charCodeAt(b)));
                if (bid >= 0) ids.push(bid);
            }
        }
        return ids;
    }

    /** Raw BPE encode of a span — no special-token recognition. */
    private function encodeBPE(text:String):Array<Int> {
        if (text.length == 0) return [];
        if (spm) return encodeSPM(text);
        var pieces:Array<String>;
        if (byteLevel) {
            pieces = byteEncodePieces(text);
        } else {
            pieces = [];
            for (i in 0...text.length) pieces.push(text.charAt(i));
        }

        // O(n log n). The previous version rescanned every adjacent pair to
        // find the best merge and REBUILT `pieces` for each one — O(n) work x
        // O(n) merges, plus an allocation per merge. Measured 4.07x per
        // doubling of the prompt: 19.8 s for 2865 tokens, ~43 min extrapolated
        // to a 32k context, i.e. the tokenizer was the long-context wall.
        //
        // Ordering must reproduce the old scan EXACTLY: lowest rank wins, ties
        // go to the LEFTMOST pair. Slots never move — a merge retires the right
        // half — so slot index is left-to-right order and (rank, slot) ordering
        // is the same tie-break.
        var n = pieces.length;
        if (n > 1) {
            var prevIx:Array<Int> = [];
            var nextIx:Array<Int> = [];
            var alive:Array<Bool> = [];
            for (i in 0...n) {
                prevIx.push(i - 1);
                nextIx.push(i + 1 < n ? i + 1 : -1);
                alive.push(true);
            }
            var hr:Array<Int> = [];
            var hi:Array<Int> = [];
            var hj:Array<Int> = [];
            for (i in 0...(n - 1)) {
                var r0 = rankOf(pieces[i], pieces[i + 1]);
                if (r0 >= 0) heapPush(hr, hi, hj, r0, i, i + 1);
            }
            while (hr.length > 0) {
                var r = hr[0];
                var a = hi[0];
                var b = hj[0];
                heapPop(hr, hi, hj);
                // Stale: a half already merged away, no longer adjacent, or one
                // side's text changed so this rank is no longer what it scores.
                if (!alive[a] || !alive[b]) continue;
                if (nextIx[a] != b) continue;
                if (rankOf(pieces[a], pieces[b]) != r) continue;
                pieces[a] = merges[r].merged;
                alive[b] = false;
                var bn = nextIx[b];
                nextIx[a] = bn;
                if (bn >= 0) prevIx[bn] = a;
                // `a` changed, so both pairs touching it are new candidates.
                var ap = prevIx[a];
                if (ap >= 0) {
                    var rl = rankOf(pieces[ap], pieces[a]);
                    if (rl >= 0) heapPush(hr, hi, hj, rl, ap, a);
                }
                if (bn >= 0) {
                    var rt = rankOf(pieces[a], pieces[bn]);
                    if (rt >= 0) heapPush(hr, hi, hj, rt, a, bn);
                }
            }
            // Walk surviving slots in order. Slot 0 never dies: a merge always
            // retires the RIGHT half, whose slot index is the larger one.
            var out:Array<String> = [];
            var cur = 0;
            while (cur >= 0) {
                out.push(pieces[cur]);
                cur = nextIx[cur];
            }
            pieces = out;
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

    /**
     * The raw byte-alphabet piece for a single token id (the per-token
     * string BEFORE `byteDecode`), or "" for an out-of-range id.
     *
     * Pairs with `decodeBuffer` for O(N) streaming decode: append each new
     * token's piece to a `StringBuf`, then `decodeBuffer(buf.toString())`.
     * `decode(ids)` would re-concatenate the whole id list every call,
     * which is O(N²) per call (O(N³) over a generation) — the dominant
     * allocation churn on long streams.
     */
    public function decodePiece(id:Int):String {
        return (id >= 0 && id < vocab.size()) ? vocab.get(id) : "";
    }

    /**
     * Finish a decode from accumulated byte-alphabet pieces: applies the
     * byte-level → UTF-8 translation when `byteLevel`, else returns as-is.
     * Must run on the FULL accumulated buffer (a multi-byte UTF-8 codepoint
     * can straddle a token boundary), so callers pass `buf.toString()`.
     */
    public function decodeBuffer(raw:String):String {
        if (spm) return spmDecodeToBytes(raw).toString();
        return byteLevel ? byteDecodeToBytes(raw, byteDecoderArrayCached()).toString() : raw;
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
        return byteDecodeToBytes(s, byteDecoderArray()).toString();
    }

    /** Instance reverse-map accessor, built once and cached (the per-call
        rebuild was a constant ~256-string/token churn). */
    function byteDecoderArrayCached():Array<Int> {
        if (_byteDecoderCache == null) _byteDecoderCache = byteDecoderArray();
        return _byteDecoderCache;
    }

    /** Byte index of the end of the last COMPLETE UTF-8 codepoint in `b`.
        Trailing bytes of an incomplete codepoint (one whose continuation
        bytes haven't arrived yet) are excluded — they're the carry. Matches
        `byteDecodeToBytes`'s `i + need <= n` completeness test. */
    static function lastCompleteUtf8(b:haxe.io.Bytes):Int {
        var n = b.length;
        var i = 0;
        var lastComplete = 0;
        while (i < n) {
            var b0 = b.get(i);
            var need = (b0 < 0x80) ? 1
                : ((b0 & 0xE0) == 0xC0) ? 2
                : ((b0 & 0xF0) == 0xE0) ? 3
                : ((b0 & 0xF8) == 0xF0) ? 4
                : 1;
            if (i + need <= n) {
                i += need;
                lastComplete = i;
            } else {
                break;
            }
        }
        return lastComplete;
    }

    /** Streaming decode. Given the incomplete-output-byte carry from the
        previous step (an output UTF-8 codepoint whose bytes straddle a token
        boundary, held in `carryHolder[0]`) and a new token `id`, returns the
        newly-completed decoded text (delta) and writes the new carry back.
        O(|piece|) per token, O(N) over a generation — no growing re-decode,
        unlike the old `rawPieces += piece` + full re-decode (O(N²) on wasm,
        where StringBuf has no amortised append). Concatenating every delta
        (+ a final `carryHolder[0].toString()` flush) reproduces `decodeBuffer`
        of the whole stream byte-for-byte. */
    public function decodeStreamStep(carryHolder:Array<haxe.io.Bytes>, id:Int):String {
        var carry = carryHolder[0];
        var piece = (id >= 0 && id < vocab.size()) ? vocab.get(id) : "";
        var pieceBytes = byteLevel
            ? byteDecodeToBytes(piece, byteDecoderArrayCached())
            : (spm ? spmDecodeToBytes(piece) : haxe.io.Bytes.ofString(piece));
        var cn = (carry != null) ? carry.length : 0;
        var pbLen = pieceBytes.length;
        // Build `combined = carry ++ pieceBytes` with explicit get/set (NOT
        // blit: rayzor.Bytes.blit's direction differs native-vs-wasm). Buffers
        // are tiny (carry <= 3, piece a few bytes), so the loop cost is trivial.
        var combined = haxe.io.Bytes.alloc(cn + pbLen);
        for (i in 0...cn) combined.set(i, carry.get(i));
        for (i in 0...pbLen) combined.set(cn + i, pieceBytes.get(i));
        var k = (byteLevel || spm) ? lastCompleteUtf8(combined) : combined.length;
        var delta = (k > 0) ? combined.sub(0, k).toString() : "";
        // Carry the trailing incomplete-codepoint bytes (always <= 3, bounded
        // by lastCompleteUtf8) as an OWNED COPY rather than a `sub` view into
        // `combined`: the holder array outlives this call, and an owned copy
        // can't dangle if Bytes reclamation ever frees `combined`. Returned via
        // the holder, not a struct field (wasm codegen drops a String field
        // from a returned anon struct).
        var tailLen = combined.length - k;
        var tail = haxe.io.Bytes.alloc(tailLen);
        for (i in 0...tailLen) tail.set(i, combined.get(k + i));
        carryHolder[0] = tail;
        return delta;
    }

    /** Core of `byteDecode`: byte-alphabet string → raw output BYTES, with a
        prebuilt reverse map. Returning Bytes (not String) lets the streaming
        path split at a UTF-8 codepoint boundary. */
    static function byteDecodeToBytes(s:String, rev:Array<Int>):haxe.io.Bytes {
        if (s == null || s.length == 0) return haxe.io.Bytes.alloc(0);
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
        return buf.sub(0, pos);
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
