import nue.arch.BertEmbedder;

/**
 * nue vector-search — the retrieval half of the vector-DB story (Phase 2).
 *
 * Embeds a small document corpus with the pure-Haxe BERT encoder, then ranks
 * the corpus against a query by cosine similarity and prints the top-k. This is
 * the "embed-server retrieves" side of the stack; a generator (llama-chat) would
 * consume the top-k passages as context.
 *
 *   rayzor run Main.hx -- --model model.gguf "how do cats behave?"
 *   rayzor run Main.hx -- --model model.gguf --topk 5 "space exploration"
 *
 * Model path: --model, a positional *.gguf, NUE_EMBED_MODEL, or GGUF.
 */
class Main {
    // A deliberately topic-diverse corpus so ranking quality is visible.
    static var CORPUS = [
        "Cats are independent animals that groom themselves and nap for most of the day.",
        "Dogs are loyal companions that enjoy fetch and long walks with their owners.",
        "Kittens knead soft blankets and purr when they feel safe and content.",
        "The Apollo program landed twelve astronauts on the surface of the Moon.",
        "Mars rovers analyze rock samples to search for ancient signs of water.",
        "A telescope gathers faint light from distant galaxies across the night sky.",
        "Sourdough bread rises slowly as wild yeast ferments the flour and water.",
        "Simmer the tomatoes with garlic and basil to build a rich pasta sauce.",
        "Espresso is brewed by forcing hot water through finely ground coffee beans.",
        "A compiler translates source code into machine instructions the CPU runs.",
        "Neural networks learn patterns by adjusting weights through backpropagation.",
        "Distributed databases replicate data across nodes for fault tolerance.",
        "Marathon runners pace themselves to conserve energy over long distances.",
        "The midfielder threaded a precise pass to set up the winning goal.",
    ];

    static function main() {
        var args = Sys.args();
        var path = modelPathFromEnv();
        var topk = 3;
        var queryParts:Array<String> = [];

        var i = 0;
        while (i < args.length) {
            var a = args[i];
            if ((a == "--model" || a == "--model-path") && i + 1 < args.length) {
                path = args[i + 1];
                i += 2;
            } else if (a == "--topk" && i + 1 < args.length) {
                var k = Std.parseInt(args[i + 1]);
                if (k != null && k + 0 > 0) topk = k + 0;
                i += 2;
            } else if (isEmpty(path) && looksLikeModelPath(a)) {
                path = a;
                i++;
            } else {
                queryParts.push(a);
                i++;
            }
        }

        if (isEmpty(path)) {
            Sys.println("usage: rayzor run Main.hx -- [--model model.gguf] [--topk K] \"query text\"");
            Sys.exit(1);
        }
        var query = (queryParts.length > 0) ? queryParts.join(" ") : "how do house cats behave?";
        if (topk > CORPUS.length) topk = CORPUS.length;

        trace("=== nue vector-search ===");
        var embedder = new BertEmbedder(path);
        trace("[index] embedding " + CORPUS.length + " documents (dim=" + embedder.dimension() + ")");

        var t0 = Sys.time();
        var docVecs:Array<Array<Float>> = [];
        for (doc in CORPUS) docVecs.push(embedder.embedText(doc));
        trace("[index] built in " + fmt(Sys.time() - t0) + "s");

        var qVec = embedder.embedText(query);
        trace("");
        trace("query: \"" + query + "\"");
        trace("top-" + topk + " by cosine similarity:");

        // Score every document, then select the top-k by descending cosine.
        // Manual selection (O(n·k)) rather than Array.sort with a comparator:
        // top-k needs no full sort, and it avoids an Array.sort-closure codegen bug.
        var scores:Array<Float> = [for (v in docVecs) cosine(qVec, v)];
        var used = [for (_ in 0...CORPUS.length) false];
        for (rank in 0...topk) {
            var best = -1;
            var bestScore = -2.0;
            for (idx in 0...CORPUS.length) {
                if (!used[idx] && scores[idx] > bestScore) {
                    bestScore = scores[idx];
                    best = idx;
                }
            }
            if (best < 0) break;
            used[best] = true;
            Sys.println("  " + (rank + 1) + ". cos=" + fmt(scores[best]) + "  " + CORPUS[best]);
        }
    }

    static function cosine(a:Array<Float>, b:Array<Float>):Float {
        var dot = 0.0;
        var na = 0.0;
        var nb = 0.0;
        for (j in 0...a.length) {
            dot += a[j] * b[j];
            na += a[j] * a[j];
            nb += b[j] * b[j];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    static function modelPathFromEnv():String {
        var path = Sys.getEnvOr("NUE_EMBED_MODEL", "RAYZOR_EMBED_MODEL");
        if (!isEmpty(path)) return path;
        path = Sys.getEnv("GGUF");
        if (!isEmpty(path)) return path;
        return null;
    }

    static function isEmpty(s:String):Bool {
        return s == null || s.length == 0;
    }

    static function looksLikeModelPath(s:String):Bool {
        return s.indexOf(".gguf") >= 0;
    }

    static inline function fmt(x:Float):String {
        return Std.string(Math.round(x * 10000) / 10000);
    }
}
