import nue.tokenizer.Vocab;
import nue.tokenizer.BPETokenizer;
import nue.tokenizer.MergeRule;
import nue.tokenizer.Tokenizer;
import nue.loader.GGUFLoader;
import nue.loader.SafetensorsLoader;
import nue.loader.ONNXLoader;
import nue.model.NamedTensorMap;

class Main {
    static function main() {
        trace("=== Phase 7: loaders + tokenizer surface check ===");

        // --- Vocab + BPE tokenizer ---
        var vocab = new Vocab();
        vocab.add("h");
        vocab.add("e");
        vocab.add("l");
        vocab.add("o");
        vocab.add("he");
        vocab.add("llo");
        vocab.add("hello");
        trace("vocab size = " + vocab.size());

        var merges:Array<MergeRule> = [
            new MergeRule("h", "e", "he"),
            new MergeRule("l", "o", "llo"),
            new MergeRule("he", "llo", "hello")
        ];

        var tok = new BPETokenizer(vocab, merges, true);
        trace("tokenizer vocab size = " + tok.vocabSize());

        var ids = tok.encode("hello");
        trace("encoded 'hello' to " + ids.length + " tokens");

        var dec = tok.decode(ids);
        trace("decoded back: '" + dec + "'");

        // --- NamedTensorMap quick check ---
        var ntm = new NamedTensorMap();
        trace("ntm size at construction = " + ntm.size());
        trace("ntm.exists('missing') = " + ntm.exists("missing"));

        // --- Loader instantiation (no file I/O) ---
        var gguf = new GGUFLoader();
        var st = new SafetensorsLoader();
        var onnx = new ONNXLoader();
        trace("GGUFLoader instantiable: " + (gguf != null));
        trace("SafetensorsLoader instantiable: " + (st != null));
        trace("ONNXLoader instantiable: " + (onnx != null));

        trace("=== Done ===");
    }
}
