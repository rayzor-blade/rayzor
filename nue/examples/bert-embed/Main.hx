import nue.arch.BertEmbedder;

/**
 * BERT encoder correctness harness (Phase 1). Loads a MiniLM GGUF via the
 * nue-module BertEmbedder (which keeps all model-field access inside
 * nue.arch), feeds GOLDEN WordPiece token ids (bypassing our tokenizer so
 * this isolates the encoder math), and reports cosine vs the ONNX/
 * sentence-transformers golden.
 *
 *   rayzor run Main.hx -- <model.gguf> <harness_ids.txt> <embeddings.f32.bin>
 *
 * Gate: cosine >= 0.999 (f32). A pre-norm/post-norm or eps/GELU defect
 * tanks it well below 0.9.
 */
class Main {
    static function main() {
        var args = Sys.args();
        if (args.length < 3) {
            Sys.println("usage: bert-embed <model.gguf> <ids.txt> <embeddings.f32.bin>");
            return;
        }
        // "tok" mode runs the WordPiece tokenizer test; default runs the
        // encoder cosine test. All model access stays inside nue.arch (cross-
        // module field/ctor access on nue objects corrupts layout —
        // bugs_import_xmodule_member_resolution).
        if (args[0] == "tok") {
            BertEmbedder.tokenizerTest(args[1], args[2], args[3]);
        } else {
            BertEmbedder.goldenTest(args[0], args[1], args[2]);
        }
    }
}
