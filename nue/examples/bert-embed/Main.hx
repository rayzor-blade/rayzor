import nue.arch.BertEmbedder;

/**
 * BERT sentence-embedding harness + demo (Phase 1). All model/tokenizer
 * access stays inside nue.arch (BertEmbedder); this entry only marshals
 * primitive args. Modes:
 *
 *   rayzor run Main.hx -- embed    <gguf> "some text"                 # end-to-end embed
 *   rayzor run Main.hx -- textgold <gguf> <sentences.txt> <emb.f32>   # text→embedding vs golden
 *   rayzor run Main.hx -- tok      <gguf> <sentences.txt> <ids.txt>   # tokenizer vs golden ids
 *   rayzor run Main.hx -- mask     <gguf>                            # padded==unpadded (attention mask)
 *   rayzor run Main.hx --          <gguf> <ids.txt> <emb.f32>         # encoder-only (golden ids)
 *
 * Gate: cosine >= 0.999 (f32). A pre/post-norm, eps, or tokenizer defect
 * tanks it well below 0.9.
 */
class Main {
    static function main() {
        var args = Sys.args();
        if (args.length < 2) {
            Sys.println("usage:");
            Sys.println("  embed    <gguf> \"text\"");
            Sys.println("  textgold <gguf> <sentences.txt> <embeddings.f32.bin>");
            Sys.println("  tok      <gguf> <sentences.txt> <ids.txt>");
            Sys.println("  mask     <gguf>");
            Sys.println("  <gguf> <ids.txt> <embeddings.f32.bin>   (encoder only)");
            return;
        }
        switch (args[0]) {
            case "embed":
                BertEmbedder.embedOne(args[1], args[2]);
            case "textgold":
                BertEmbedder.textGoldenTest(args[1], args[2], args[3]);
            case "tok":
                BertEmbedder.tokenizerTest(args[1], args[2], args[3]);
            case "mask":
                BertEmbedder.maskTest(args[1]);
            default:
                BertEmbedder.goldenTest(args[0], args[1], args[2]);
        }
    }
}
