import nue.loader.GGUFLoader;
import nue.model.ModelMetadata;
import nue.tokenizer.BPETokenizer;

/**
 * Phase 9 demo — real Llama-3-8B GGUF through the full loader abstraction.
 *
 * Goes through `GGUFLoader` instead of the bare `GGUFReader` pattern that
 * `llama-smoke/Main.hx` uses, so this exercises the cross-file class
 * dispatch path that was previously broken by the parser's missing
 * support for `case A, B:` (Haxe multi-value case clauses). Fixed in
 * the parser; see `bugs_cross_module_dispatch_complexity.md`.
 *
 * Reads metadata, builds the BPE tokenizer, dumps vocab + special-token
 * ids, and round-trips a real prompt through encode/decode. The
 * transformer forward pass is the next pending piece.
 */
class Main {
    static function main() {
        var path = "/Users/amaterasu/.cache/huggingface/hub/models--lmstudio-community--Meta-Llama-3-8B-Instruct-GGUF/snapshots/0910a3e69201d274d4fd68e89448114cd78e4c82/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";

        trace("=== nue llama-chat (via GGUFLoader) ===");
        trace("model: " + path);
        trace("");

        var loader = new GGUFLoader();
        var meta = loader.readMetadata(path);

        trace("[meta] arch     = " + meta.architecture);
        trace("[meta] hidden   = " + meta.hiddenSize);
        trace("[meta] layers   = " + meta.numLayers);
        trace("[meta] heads    = " + meta.numHeads);
        trace("[meta] kv_heads = " + meta.numKvHeads);
        trace("[meta] vocab    = " + meta.vocabSize);
        trace("[meta] ctx      = " + meta.maxSeqLen);
        trace("[meta] ffn      = " + meta.intermediateSize);
        trace("[meta] norm_eps = " + meta.normEps);
        trace("[meta] rope     = " + meta.ropeBase);
        trace("[meta] tie_emb  = " + meta.tieWordEmbeddings);
        trace("");

        var tok = loader.tokenizer(path);
        trace("[tok] vocab   = " + tok.vocabSize());
        trace("[tok] bos id  = " + tok.specialId("<|begin_of_text|>"));
        trace("[tok] eos id  = " + tok.specialId("<|end_of_text|>"));

        var bpe = cast(tok, BPETokenizer);
        var prompt = "Hello";
        var ids = bpe.encode(prompt);
        trace("[encode] '" + prompt + "' -> " + ids.length + " tokens");
        trace("[decode] round-trip = '" + bpe.decode(ids) + "'");

        trace("");
        trace("=== Done ===");
    }
}
