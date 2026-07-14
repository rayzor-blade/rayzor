package nue.tokenizer;

import haxe.ds.StringMap;

/**
 * WordPiece tokenizer — the BERT / DistilBERT / MiniLM / BGE scheme.
 *
 * Reproduces the HuggingFace `BertNormalizer` + `BertPreTokenizer` +
 * `WordPiece` pipeline so token ids byte-match `transformers`:
 *
 *   1. clean_text        — drop control chars, normalize whitespace to space
 *   2. chinese_chars     — surround CJK codepoints with spaces (each is a token)
 *   3. strip_accents     — NFD-style: drop combining marks (Latin-1 precomposed
 *                          letters mapped to their base; other scripts pass through)
 *   4. lowercase         — ASCII + Latin-1 lowercase
 *   5. pre-tokenize      — split on whitespace, isolate each punctuation char
 *   6. wordpiece         — greedy longest-prefix match, `##` continuation,
 *                          `[UNK]` fallback, max 100 codepoints/word
 *
 * rayzor `String` is UTF-8 bytes, so this decodes to codepoints itself.
 * Full Unicode NFD isn't implemented — accent stripping covers Latin-1
 * Supplement (café, naïve, résumé, …); other combining scripts pass
 * through unchanged, which is correct for the common embedding corpora.
 */
class WordPieceTokenizer implements Tokenizer {
    public var vocab:Vocab;
    public var specials:StringMap<Int>;
    public var specialNames:Array<String>;
    var unkId:Int;
    var unkToken:String;
    var wordStartPrefix:String;
    var contPrefix:String;
    var maxChars:Int;

    public function new(vocab:Vocab, unkToken:String = "[UNK]") {
        this.vocab = vocab;
        this.specials = new StringMap<Int>();
        this.specialNames = [];
        this.unkToken = unkToken;
        this.unkId = vocab.lookup(unkToken);
        this.maxChars = 100;

        // Two vocab encodings map to the same ids: HF WordPiece marks word-start
        // pieces bare and continuations with `##`; llama.cpp's bert GGUF marks
        // word-start with `▁` (U+2581) and continuations bare. Detect which by
        // probing for a `▁`-prefixed common word.
        if (vocab.lookup("▁the") >= 0 || vocab.lookup("▁a") >= 0) {
            this.wordStartPrefix = "▁"; // ▁
            this.contPrefix = "";
        } else {
            this.wordStartPrefix = "";
            this.contPrefix = "##";
        }
    }

    public function addSpecial(name:String, id:Int):Void {
        specials.set(name, id);
        specialNames.push(name);
    }

    public function specialId(name:String):Int {
        return specials.exists(name) ? specials.get(name) : -1;
    }

    public function vocabSize():Int {
        return vocab.tokens.length;
    }

    // --- encode ---

    public function encode(text:String):Array<Int> {
        var cps = decodeUtf8(text);
        var norm = normalize(cps);            // cleaned/accent-stripped/lowercased, CJK space-padded
        var words = splitWordsAndPunct(norm); // Array<Array<Int>> — pre-tokenized pieces
        var ids:Array<Int> = [];
        for (w in words) {
            wordPiece(w, ids);
        }
        return ids;
    }

    /** Encode with the `[CLS] ... [SEP]` wrapping BERT expects. */
    public function encodeWithSpecials(text:String):Array<Int> {
        var cls = specialId("[CLS]");
        var sep = specialId("[SEP]");
        var body = encode(text);
        var out:Array<Int> = [];
        if (cls >= 0) out.push(cls);
        for (id in body) out.push(id);
        if (sep >= 0) out.push(sep);
        return out;
    }

    // --- normalization ---

    function normalize(cps:Array<Int>):Array<Int> {
        var out:Array<Int> = [];
        for (cp in cps) {
            // clean_text: drop NUL / replacement / control chars
            if (cp == 0 || cp == 0xFFFD || isControl(cp)) continue;
            if (isWhitespace(cp)) {
                out.push(0x20);
                continue;
            }
            // strip_accents (Latin-1 precomposed → base) then lowercase
            var low = toLower(stripAccent(cp));
            if (isCjk(low)) {
                // handle_chinese_chars: each CJK char is its own token
                out.push(0x20);
                out.push(low);
                out.push(0x20);
            } else {
                out.push(low);
            }
        }
        return out;
    }

    // --- pre-tokenization: whitespace split + punctuation isolation ---

    function splitWordsAndPunct(cps:Array<Int>):Array<Array<Int>> {
        var words:Array<Array<Int>> = [];
        var cur:Array<Int> = [];
        for (cp in cps) {
            if (cp == 0x20) {
                if (cur.length > 0) { words.push(cur); cur = []; }
            } else if (isPunct(cp)) {
                if (cur.length > 0) { words.push(cur); cur = []; }
                words.push([cp]);
            } else {
                cur.push(cp);
            }
        }
        if (cur.length > 0) words.push(cur);
        return words;
    }

    // --- WordPiece greedy longest-prefix ---

    function wordPiece(word:Array<Int>, out:Array<Int>):Void {
        var n = word.length;
        if (n == 0) return;
        if (n > maxChars) {
            out.push(unkId);
            return;
        }
        var start = 0;
        var subTokens:Array<Int> = [];
        var bad = false;
        while (start < n) {
            var end = n;
            var curId = -1;
            while (end > start) {
                var prefix = (start == 0) ? wordStartPrefix : contPrefix;
                var piece = prefix + encodeUtf8(word, start, end);
                var id = vocab.lookup(piece);
                if (id >= 0) { curId = id; break; }
                end--;
            }
            if (curId < 0) { bad = true; break; }
            subTokens.push(curId);
            start = end;
        }
        if (bad) {
            out.push(unkId);
        } else {
            for (id in subTokens) out.push(id);
        }
    }

    // --- UTF-8 codec ---

    static function decodeUtf8(s:String):Array<Int> {
        var out:Array<Int> = [];
        var i = 0;
        var n = s.length;
        while (i < n) {
            var b0 = s.charCodeAt(i);
            if (b0 < 0x80) {
                out.push(b0);
                i += 1;
            } else if (b0 < 0xE0) {
                var b1 = (i + 1 < n) ? s.charCodeAt(i + 1) : 0;
                out.push(((b0 & 0x1F) << 6) | (b1 & 0x3F));
                i += 2;
            } else if (b0 < 0xF0) {
                var b1 = (i + 1 < n) ? s.charCodeAt(i + 1) : 0;
                var b2 = (i + 2 < n) ? s.charCodeAt(i + 2) : 0;
                out.push(((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F));
                i += 3;
            } else {
                var b1 = (i + 1 < n) ? s.charCodeAt(i + 1) : 0;
                var b2 = (i + 2 < n) ? s.charCodeAt(i + 2) : 0;
                var b3 = (i + 3 < n) ? s.charCodeAt(i + 3) : 0;
                out.push(((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F));
                i += 4;
            }
        }
        return out;
    }

    static function encodeUtf8(cps:Array<Int>, start:Int, end:Int):String {
        // `StringBuf.addChar` takes a codepoint and UTF-8-encodes it, so pass
        // the codepoint directly — a manual byte split would be re-encoded
        // (double-encoding) for anything above ASCII.
        var buf = new StringBuf();
        for (k in start...end) buf.addChar(cps[k]);
        return buf.toString();
    }

    // --- codepoint classification ---

    static inline function isWhitespace(cp:Int):Bool {
        // ASCII space/tab/newline/CR + Unicode spaces treated as separators.
        return cp == 0x20 || cp == 0x09 || cp == 0x0A || cp == 0x0D
            || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A)
            || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
    }

    static inline function isControl(cp:Int):Bool {
        // \t \n \r already handled as whitespace by the caller. Treat other
        // C0/C1 controls + format chars as removable.
        if (cp == 0x09 || cp == 0x0A || cp == 0x0D) return false;
        return (cp < 0x20) || (cp >= 0x7F && cp <= 0x9F);
    }

    static inline function isCjk(cp:Int):Bool {
        return (cp >= 0x4E00 && cp <= 0x9FFF)
            || (cp >= 0x3400 && cp <= 0x4DBF)
            || (cp >= 0x20000 && cp <= 0x2A6DF)
            || (cp >= 0x2A700 && cp <= 0x2B73F)
            || (cp >= 0x2B740 && cp <= 0x2B81F)
            || (cp >= 0x2B820 && cp <= 0x2CEAF)
            || (cp >= 0xF900 && cp <= 0xFAFF)
            || (cp >= 0x2F800 && cp <= 0x2FA1F);
    }

    static inline function isPunct(cp:Int):Bool {
        // BERT: ASCII punctuation ranges + any Unicode P* category. We cover
        // the ASCII ranges (exhaustive for typical text) and common Unicode
        // punctuation blocks.
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64)
            || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
        // General Punctuation + CJK punctuation blocks (approximate P*).
        return (cp >= 0x2000 && cp <= 0x206F && !isWhitespace(cp))
            || (cp >= 0x3000 && cp <= 0x303F && cp != 0x3000)
            || (cp >= 0xFF00 && cp <= 0xFF0F);
    }

    static inline function toLower(cp:Int):Int {
        if (cp >= 0x41 && cp <= 0x5A) return cp + 32;          // ASCII A-Z
        if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) return cp + 32; // Latin-1 uppercase
        return cp;
    }

    /**
     * Latin-1 Supplement precomposed letter → base ASCII (NFD then drop the
     * combining mark). Letters that do NOT decompose under NFD (Æ Ð Ø Þ ß æ ð
     * ø þ) pass through unchanged, matching HF. Case is preserved here; the
     * caller lowercases afterwards.
     */
    static function stripAccent(cp:Int):Int {
        // uppercase
        if (cp >= 0xC0 && cp <= 0xC5) return 0x41; // À-Å → A
        if (cp == 0xC7) return 0x43;               // Ç → C
        if (cp >= 0xC8 && cp <= 0xCB) return 0x45; // È-Ë → E
        if (cp >= 0xCC && cp <= 0xCF) return 0x49; // Ì-Ï → I
        if (cp == 0xD1) return 0x4E;               // Ñ → N
        if (cp >= 0xD2 && cp <= 0xD6) return 0x4F; // Ò-Ö → O
        if (cp >= 0xD9 && cp <= 0xDC) return 0x55; // Ù-Ü → U
        if (cp == 0xDD) return 0x59;               // Ý → Y
        // lowercase
        if (cp >= 0xE0 && cp <= 0xE5) return 0x61; // à-å → a
        if (cp == 0xE7) return 0x63;               // ç → c
        if (cp >= 0xE8 && cp <= 0xEB) return 0x65; // è-ë → e
        if (cp >= 0xEC && cp <= 0xEF) return 0x69; // ì-ï → i
        if (cp == 0xF1) return 0x6E;               // ñ → n
        if (cp >= 0xF2 && cp <= 0xF6) return 0x6F; // ò-ö → o
        if (cp >= 0xF9 && cp <= 0xFC) return 0x75; // ù-ü → u
        if (cp == 0xFD || cp == 0xFF) return 0x79; // ý ÿ → y
        return cp;
    }

    // --- decode (id → text), for round-tripping / debugging ---

    public function decode(ids:Array<Int>):String {
        var buf = new StringBuf();
        for (k in 0...ids.length) {
            var id = ids[k];
            if (id < 0 || id >= vocab.tokens.length) continue;
            var tok = vocab.tokens[id];
            if (StringTools.startsWith(tok, "##")) {
                buf.add(tok.substr(2));
            } else {
                if (k > 0) buf.add(" ");
                buf.add(tok);
            }
        }
        return buf.toString();
    }

    public function decodePiece(id:Int):String {
        if (id < 0 || id >= vocab.tokens.length) return "";
        return vocab.tokens[id];
    }

    public function decodeBuffer(raw:String):String {
        return raw;
    }

    public function decodeStreamStep(carryHolder:Array<haxe.io.Bytes>, id:Int):String {
        return decodePiece(id);
    }
}
