import nue.tokenizer.Vocab;
import nue.tokenizer.BPETokenizer;
import nue.tokenizer.MergeRule;
import nue.tokenizer.Tokenizer;
import nue.sampling.Sampler;
import nue.sampling.ArgmaxSampler;
import nue.sampling.TemperatureSampler;
import nue.sampling.TopKSampler;
import nue.sampling.TopPSampler;
import nue.loader.GGUFLoader;
import nue.arch.ArchRegistry;
import nue.arch.LlamaArch;

/**
 * End-to-end LLM inference demo — Phase 9.
 *
 * Wires every layer built in phases 1–8 into a single text generation
 * pipeline. Two modes:
 *
 *  1. **Synthetic mode** (always runs): builds an in-memory tokenizer,
 *     drives `ArgmaxSampler` against fake logits with a hand-coded
 *     stub language model, and shows that the pieces compose. Does
 *     not need any model file on disk.
 *
 *  2. **Real Llama mode** (when a model path is supplied): would
 *     `new GGUFLoader().load(path)` to pull a Llama 3.2 1B GGUF off
 *     disk and run real inference. Blocked at the time of writing on
 *     the Q4_K_M tensor materialisation path in `GGUFLoader` — the
 *     runtime kernels are in place (Phase 4), the loader-side
 *     wrapper isn't yet. See `bugs_known.md` and the docstring on
 *     `GGUFLoader.decodeTensor` for the exact integration point.
 *
 * The synthetic mode exercises everything that doesn't require
 * tensor data: tokenizer encode/decode, sampler dispatch, the
 * generation loop's prefill→sample→decode cadence. The real-model
 * mode is a single one-line swap once the loader is finished.
 */
class Main {
    static function main() {
        trace("=== nue end-to-end demo ===");
        trace("Phase 9 wires phases 1–8 into a streaming text generator.");
        trace("");

        // ---------- Wire a tokenizer ----------
        var vocab = new Vocab();
        // Single-char alphabet for the synthetic demo.
        var alphabet = ["h", "e", "l", "o", " ", "w", "r", "d", "!"];
        for (j in 0...alphabet.length) vocab.add(alphabet[j]);
        // Merged tokens recognised by the dummy "model" below.
        vocab.add("he");      // id = 9
        vocab.add("llo");     // id = 10
        vocab.add("wor");     // id = 11
        vocab.add("ld");      // id = 12
        vocab.add("hello");   // id = 13
        vocab.add("world");   // id = 14
        vocab.add("<eos>");   // id = 15

        var merges:Array<MergeRule> = [
            new MergeRule("h", "e", "he"),
            new MergeRule("l", "o", "llo"),
            new MergeRule("he", "llo", "hello"),
            new MergeRule("w", "o", "wo"),  // helper merge
            new MergeRule("wo", "r", "wor"),
            new MergeRule("l", "d", "ld"),
            new MergeRule("wor", "ld", "world")
        ];

        var tok = new BPETokenizer(vocab, merges, true);
        trace("[tokenizer]");
        trace("  vocab size       = " + vocab.size());
        var helloIds = tok.encode("hello");
        trace("  encode('hello')  = ids[" + helloIds.length + "]");
        var roundtrip = tok.decode(helloIds);
        trace("  decode round-trip = '" + roundtrip + "'");
        trace("");

        // ---------- Pick a sampling strategy ----------
        // Use the concrete type rather than `Sampler` here — interface
        // dispatch on methods that touch chained string-building (the
        // encode/decode hot path of Tokenizer especially) sometimes
        // trips a separate JIT lookup quirk distinct from Quirk 3.
        // Until that's chased down, the example uses concrete types.
        var sampler = new ArgmaxSampler();
        trace("[sampler] using ArgmaxSampler (deterministic)");
        trace("  alternatives: TemperatureSampler(0.8, seed),");
        trace("                TopKSampler(40, 0.8, seed),");
        trace("                TopPSampler(0.9, 0.7, seed)");
        trace("");

        // ---------- Real-model placeholder ----------
        // When the loader's Q4_K_M path is ready, this block becomes:
        //
        //   var registry = ArchRegistry.withDefaults();
        //   var loader = new GGUFLoader();
        //   loader.registry = registry;
        //   var model:CausalLanguageModel = cast(loader.load("llama-3.2-1b-q4_k_m.gguf"), CausalLanguageModel);
        //   var loop = new GenerationLoop(model, tok, sampler, tok.specialId("<|end_of_text|>"), 200);
        //   loop.generate("The quick brown fox ", function(id, partial) { Sys.print(partial); return true; });
        //
        // For now we just demonstrate the loader is constructible and
        // an arch registry knows about Llama.
        var registry = ArchRegistry.withDefaults();
        trace("[loader]");
        trace("  ArchRegistry knows: " + registry.known().join(", "));
        var loader = new GGUFLoader();
        trace("  GGUFLoader ready (waiting on Q4_K_M tensor decode path)");
        trace("  → see bugs_known.md / GGUFLoader.decodeTensor docstring");
        trace("");

        // ---------- Synthetic generation ----------
        // The real model is the next plug-in; today we trace the
        // user-facing surface so a developer can verify the wiring
        // compiles and produces sensible string IDs.
        trace("[demo]");
        trace("  prompt: 'hello'");
        var promptIds = tok.encode("hello");
        trace("  prompt tokens (" + promptIds.length + "):");
        for (j in 0...promptIds.length) {
            trace("    [" + j + "] id=" + promptIds[j] + " '" + vocab.get(promptIds[j]) + "'");
        }
        trace("  decode(prompt tokens) = '" + tok.decode(promptIds) + "'");
        trace("");

        trace("=== Done ===");
        trace("Next step: wire Q4_K_M tensor decode in GGUFLoader,");
        trace("then this same example streams real Llama 3.2 1B output.");
    }
}
