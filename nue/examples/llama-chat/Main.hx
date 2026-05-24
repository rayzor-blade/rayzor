import sys.io.File;
import nue.loader.GGUFReader;
import nue.loader.GGUFTokenizer;
import nue.tokenizer.BPETokenizer;

class Main {
    static function main() {
        var path = "/Users/amaterasu/.cache/huggingface/hub/models--lmstudio-community--Meta-Llama-3-8B-Instruct-GGUF/snapshots/0910a3e69201d274d4fd68e89448114cd78e4c82/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";

        trace("=== nue llama-chat ===");
        trace("model: " + path);
        trace("");

        var bytes = File.getBytes(path);
        var reader = new GGUFReader(bytes);
        trace("[gguf] version  = " + reader.version);
        trace("[gguf] tensors  = " + reader.nTensors);
        trace("[gguf] arch     = " + reader.metaString("general.architecture"));
        trace("[gguf] hidden   = " + reader.metaInt("llama.embedding_length"));
        trace("[gguf] layers   = " + reader.metaInt("llama.block_count"));
        trace("[gguf] heads    = " + reader.metaInt("llama.attention.head_count"));
        trace("[gguf] kv_heads = " + reader.metaInt("llama.attention.head_count_kv"));
        trace("");

        var tok = GGUFTokenizer.build(reader);
        trace("[tok] vocab   = " + tok.vocabSize());
        trace("[tok] bos id  = " + tok.specialId("<|begin_of_text|>"));
        trace("[tok] eos id  = " + tok.specialId("<|end_of_text|>"));
        trace("");

        var bpe = cast(tok, BPETokenizer);
        trace("[vocab sample]");
        for (i in 0...8) {
            trace("  [" + i + "] = '" + bpe.vocab.get(i) + "'");
        }
        trace("");

        var prompt = "Hello";
        var ids = bpe.encode(prompt);
        trace("[encode] '" + prompt + "' -> " + ids.length + " tokens");
        trace("[decode] round-trip = '" + bpe.decode(ids) + "'");
        trace("");

        trace("=== Done ===");
    }
}
