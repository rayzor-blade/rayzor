// Quality eval harness: teacher-forced perplexity, and an A/B mode that
// reports how often two configurations pick the same next token.
//
// Why it exists: quality decisions were being made by reading two generated
// samples and judging whether both "looked coherent". That cannot distinguish
// a kernel that is slightly lossy from one that is broken, and it blocked
// NUE_FLASH_BATCH from being defaulted on even after its arithmetic was proven
// bit-identical to the shipping decode path (nue/tests/flashbatch).
//
//   nue-eval <model.gguf> <text-file> [--chunk N] [--max-chunks N] [--dump F]
//
// Perplexity is computed the way llama.cpp does it: split the token stream
// into chunks of `chunk` tokens, forward each chunk from an empty cache, and
// accumulate -log p(token[i+1] | token[0..i]) over the positions where a next
// token exists. Chunked-from-empty is deliberate -- it drives the PREFILL
// path, which is where the batched attention kernel actually runs. A
// token-at-a-time loop would exercise decode only and measure nothing about
// prefill.
//
// --dump writes one line per position: `pos<TAB>argmax<TAB>nll`. Diff two
// dumps to get exact top-1 agreement between configurations, which is the
// number a default-on decision should rest on.
import nue.loader.GGUFLoader;
import nue.arch.LlamaModel;
import nue.tokenizer.BPETokenizer;
import rayzor.ds.Tensor;
import sys.io.File;

class Main {
    static function fmt(v:Float, dp:Int = 4):String {
        var m = Math.pow(10, dp);
        return Std.string(Math.round(v * m) / m);
    }

    // log(sum(exp(x))) over one logit row, in a numerically safe form: the max
    // is factored out so exp() never sees a large positive argument.
    static function logSumExp(t:Tensor, base:Int, vocab:Int):Float {
        var mx = -1e30;
        for (v in 0...vocab) {
            var x = t.getFlat(base + v);
            if (x > mx) mx = x;
        }
        var acc = 0.0;
        for (v in 0...vocab) acc += Math.exp(t.getFlat(base + v) - mx);
        return mx + Math.log(acc);
    }

    static function argmax(t:Tensor, base:Int, vocab:Int):Int {
        var best = 0;
        var bv = -1e30;
        for (v in 0...vocab) {
            var x = t.getFlat(base + v);
            if (x > bv) { bv = x; best = v; }
        }
        return best;
    }

    static function usage():Void {
        Sys.println("usage: nue-eval <model.gguf> <text-file> [--chunk N] [--max-chunks N] [--dump FILE]");
        Sys.println("  --chunk       tokens per forward (default 512)");
        Sys.println("  --max-chunks  stop after N chunks (default 0 = all)");
        Sys.println("  --dump FILE   write pos/argmax/nll per position for an A/B diff");
    }

    static function main() {
        var args = Sys.args();
        if (args.length < 2) { usage(); Sys.exit(2); }
        var modelPath = args[0];
        var textPath = args[1];
        var chunk = 512;
        var maxChunks = 0;
        var dumpPath:String = null;
        var i = 2;
        while (i < args.length) {
            switch (args[i]) {
                case "--chunk": chunk = Std.parseInt(args[i + 1]); i += 2;
                case "--max-chunks": maxChunks = Std.parseInt(args[i + 1]); i += 2;
                case "--dump": dumpPath = args[i + 1]; i += 2;
                default: Sys.println("unknown argument: " + args[i]); usage(); Sys.exit(2);
            }
        }
        if (chunk == null || chunk < 2) { Sys.println("--chunk must be >= 2"); Sys.exit(2); }

        if (!sys.FileSystem.exists(modelPath)) {
            Sys.println("error: model not found: " + modelPath); Sys.exit(2);
        }
        if (!sys.FileSystem.exists(textPath)) {
            Sys.println("error: text not found: " + textPath); Sys.exit(2);
        }

        var text = File.getContent(textPath);
        var loader = new GGUFLoader();
        var loaded = loader.loadWithTokenizer(modelPath, chunk + 8);
        var tok = cast(loaded.tokenizer, BPETokenizer);
        var model = cast(loaded.model, LlamaModel);
        var vocab = loaded.metadata.vocabSize;

        var ids = tok.encode(text);
        Sys.println("[eval] model=" + modelPath);
        Sys.println("[eval] tokens=" + ids.length + " chunk=" + chunk
            + " vocab=" + vocab);
        if (ids.length < 2) { Sys.println("error: need at least 2 tokens"); Sys.exit(2); }

        // Accumulated and written with File.saveContent at the end:
        // FileOutput.writeString is a no-op on this runtime (the file is
        // created and stays empty, with no error), so a streaming dump would
        // silently produce nothing to diff.
        var dumpLines:Array<String> = dumpPath != null ? [] : null;

        var totalNll = 0.0;
        var counted = 0;
        var chunks = 0;
        var pos = 0;
        var t0 = Sys.time();

        while (pos + 1 < ids.length) {
            var n = chunk;
            if (pos + n > ids.length) n = ids.length - pos;
            if (n < 2) break;

            var window = ids.slice(pos, pos + n);
            model.resetCache();
            var logits = model.forwardIds(window);
            var shp = logits.shape();
            var rows = shp[0];

            // The per-op path returns [seq, vocab]; the fused CoreML prefill
            // returns only the last row. Perplexity needs every position, so
            // refuse rather than silently score one token per chunk.
            if (rows < n - 1) {
                Sys.println("error: forwardIds returned " + rows + " row(s) for a "
                    + n + "-token chunk. This build reduces prefill to the last"
                    + " row (fused graph prefill); disable it for eval.");
                logits.free();
                Sys.exit(3);
            }

            for (r in 0...(n - 1)) {
                var target = window[r + 1];
                var base = r * vocab;
                var lse = logSumExp(logits, base, vocab);
                var nll = lse - logits.getFlat(base + target);
                totalNll += nll;
                counted++;
                if (dumpLines != null) {
                    dumpLines.push((pos + r) + "\t" + argmax(logits, base, vocab)
                        + "\t" + fmt(nll, 6));
                }
            }
            logits.free();

            chunks++;
            var ppl = Math.exp(totalNll / counted);
            Sys.println("[chunk " + chunks + "] pos=" + pos + " scored=" + counted
                + " ppl=" + fmt(ppl));
            if (maxChunks > 0 && chunks >= maxChunks) break;
            pos += n;
        }

        if (dumpLines != null) {
            File.saveContent(dumpPath, dumpLines.join("\n") + "\n");
            Sys.println("[eval] dump -> " + dumpPath + " (" + dumpLines.length + " rows)");
        }

        if (counted == 0) {
            Sys.println("error: scored no positions");
            Sys.exit(3);
        }
        var ppl = Math.exp(totalNll / counted);
        Sys.println("[eval] positions=" + counted + " chunks=" + chunks
            + " elapsed=" + fmt(Sys.time() - t0, 2) + "s");
        Sys.println("PPL " + fmt(ppl));
        Sys.println("NLL " + fmt(totalNll / counted, 6));
    }
}
