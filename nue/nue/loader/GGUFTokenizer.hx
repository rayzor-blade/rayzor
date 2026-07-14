package nue.loader;

import nue.loader.GGUFReader.MetaValue;
import nue.tokenizer.Tokenizer;
import nue.tokenizer.Vocab;
import nue.tokenizer.MergeRule;
import nue.tokenizer.BPETokenizer;
import nue.tokenizer.WordPieceTokenizer;

/**
 * Build a `BPETokenizer` from the metadata table of a parsed
 * `GGUFReader`. The GGUF v3 tokenizer convention places everything
 * under the `tokenizer.ggml.*` namespace:
 *
 * ```
 *   tokenizer.ggml.model          : String   ("llama" | "gpt2")
 *   tokenizer.ggml.pre            : String   ("default" | "llama3" | ...)
 *   tokenizer.ggml.tokens         : Array<String>  (token strings, index = id)
 *   tokenizer.ggml.scores         : Array<Float>   (optional, SentencePiece)
 *   tokenizer.ggml.merges         : Array<String>  ("left right" per entry)
 *   tokenizer.ggml.bos_token_id   : Int
 *   tokenizer.ggml.eos_token_id   : Int
 *   tokenizer.ggml.padding_token_id    : Int  (optional)
 *   tokenizer.ggml.unknown_token_id    : Int  (optional)
 * ```
 *
 * The extracted `BPETokenizer` is wired with byte-level mode when the
 * model is `gpt2` (tiktoken/Llama-3 style) and SentencePiece-flavoured
 * scores when the model is `llama` (Llama-1/2/Mistral).
 *
 * **Limitations (v1).**
 *   - Pre-tokenization regex isn't applied yet; `encode()` runs the
 *     simple per-character split and BPE merges. Outputs from `encode`
 *     therefore won't byte-match what Llama 3 expects — use the
 *     extracted tokenizer's `decode()` and feed model IDs directly
 *     until pre-tokenization lands.
 *   - Token-type metadata is ignored. All tokens are treated as
 *     regular vocabulary entries.
 */
class GGUFTokenizer {
    /**
     * Build a WordPiece tokenizer (BERT / RoBERTa / MiniLM / BGE) from a
     * GGUF whose `tokenizer.ggml.model == "bert"`. The vocab list is the
     * same `tokenizer.ggml.tokens` array; there are no merges. BERT specials
     * live as ordinary vocab entries, registered by their canonical names.
     */
    public static function buildWordPiece(reader:GGUFReader):WordPieceTokenizer {
        var tokens = readStringArray(reader, "tokenizer.ggml.tokens");
        if (tokens == null || tokens.length == 0) {
            throw "GGUFTokenizer: missing 'tokenizer.ggml.tokens' — file does not embed a vocabulary.";
        }
        var vocab = new Vocab();
        for (i in 0...tokens.length) vocab.addWithScore(tokens[i], 0.0);

        var tok = new WordPieceTokenizer(vocab, "[UNK]");
        var specials = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"];
        for (i in 0...specials.length) {
            var id = vocab.lookup(specials[i]);
            if (id >= 0) tok.addSpecial(specials[i], id);
        }
        return tok;
    }

    public static function build(reader:GGUFReader):Tokenizer {
        var modelType = readStringOr(reader, "tokenizer.ggml.model", "gpt2");
        // BERT-family GGUFs carry a WordPiece vocab with no merges; the BPE
        // path below would build a merge-less tokenizer that mis-segments
        // every multi-piece word. Dispatch on the model tag so a single
        // `build`/`loadWithTokenizer` returns the right scheme per file.
        if (modelType == "bert") return buildWordPiece(reader);
        var byteLevel = (modelType == "gpt2");

        var tokens = readStringArray(reader, "tokenizer.ggml.tokens");
        if (tokens == null || tokens.length == 0) {
            throw "GGUFTokenizer: missing 'tokenizer.ggml.tokens' — file does not embed a vocabulary.";
        }

        var scores = readFloatArrayOr(reader, "tokenizer.ggml.scores");
        var vocab = new Vocab();
        for (i in 0...tokens.length) {
            var score = (scores != null && i < scores.length) ? scores[i] : 0.0;
            vocab.addWithScore(tokens[i], score);
        }

        var mergeStrings = readStringArrayOr(reader, "tokenizer.ggml.merges");
        var merges:Array<MergeRule> = [];
        if (mergeStrings != null) {
            for (m in mergeStrings) {
                var rule = parseMerge(m);
                if (rule != null) merges.push(rule);
            }
        }

        var tok = new BPETokenizer(vocab, merges, byteLevel);

        // Register the standard special-token IDs when present. The
        // string keys here mirror what Llama-3 / Mistral chat templates
        // emit so callers can look them up by familiar name.
        registerSpecial(tok, reader, "tokenizer.ggml.bos_token_id", "<|begin_of_text|>");
        registerSpecial(tok, reader, "tokenizer.ggml.eos_token_id", "<|end_of_text|>");
        registerSpecial(tok, reader, "tokenizer.ggml.padding_token_id", "<|pad|>");
        registerSpecial(tok, reader, "tokenizer.ggml.unknown_token_id", "<|unk|>");

        // Llama-3 chat-template specials live as ordinary vocab entries
        // with no dedicated GGUF metadata key. Look them up directly so
        // callers can construct chat-formatted prompts without hard-
        // coded numeric IDs.
        var chatSpecials = [
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|begin_of_text|>",
            "<|end_of_text|>"
        ];
        for (i in 0...chatSpecials.length) {
            var name = chatSpecials[i];
            var id = vocab.lookup(name);
            if (id < 0) continue;
            if (tok.specialId(name) >= 0) continue;
            tok.addSpecial(name, id);
        }

        return tok;
    }

    static function parseMerge(line:String):MergeRule {
        // Each merge entry is "<left> <right>" with a single literal
        // space separator. Tokens themselves can contain a space — but
        // only at most one separator, so split on the FIRST space.
        var sp = line.indexOf(" ");
        if (sp <= 0 || sp >= line.length - 1) return null;
        var left = line.substr(0, sp);
        var right = line.substr(sp + 1);
        return new MergeRule(left, right, left + right);
    }

    static function readStringArray(reader:GGUFReader, key:String):Array<String> {
        var items = reader.metaArray(key);
        return collectStrings(items, key);
    }

    static function readStringArrayOr(reader:GGUFReader, key:String):Array<String> {
        var items = reader.metaArrayOr(key);
        if (items == null) return null;
        return collectStrings(items, key);
    }

    static function collectStrings(items:Array<MetaValue>, key:String):Array<String> {
        var result:Array<String> = [];
        for (it in items) {
            switch (it) {
                case Str(s): result.push(s);
                case _:
                    throw "GGUFTokenizer: '" + key + "' contains a non-string element.";
            }
        }
        return result;
    }

    static function readFloatArrayOr(reader:GGUFReader, key:String):Array<Float> {
        var items = reader.metaArrayOr(key);
        if (items == null) return null;
        var result:Array<Float> = [];
        for (it in items) {
            switch (it) {
                case F32(v) | F64(v): result.push(v);
                case U8(v) | I8(v) | U16(v) | I16(v) | U32(v) | I32(v) | U64(v) | I64(v):
                    result.push(v);
                case _:
                    return null;
            }
        }
        return result;
    }

    static function readStringOr(reader:GGUFReader, key:String, fallback:String):String {
        var v = reader.findMeta(key);
        if (v == null) return fallback;
        return switch (v) {
            case Str(s): s;
            case _: fallback;
        };
    }

    static function registerSpecial(
        tok:BPETokenizer, reader:GGUFReader, key:String, name:String
    ):Void {
        var v = reader.findMeta(key);
        if (v == null) return;
        switch (v) {
            case U8(id) | I8(id) | U16(id) | I16(id) | U32(id) | I32(id) | U64(id) | I64(id):
                tok.addSpecial(name, id);
            case _:
        }
    }
}
