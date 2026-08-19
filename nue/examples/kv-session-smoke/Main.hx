import nue.loader.GGUFLoader;
import nue.tokenizer.BPETokenizer;
import nue.arch.LlamaModel;
import nue.transformer.KVSession;
import rayzor.ds.Tensor;

/**
 * Does a session actually isolate a conversation?
 *
 * Two conversations decoded one after the other are the control. The same two
 * decoded a token at a time, alternating, are the test: if the sessions shared
 * any state, each would be reading the other's keys and values and the tokens
 * would diverge. Equality across both runs is the property that makes one
 * loaded model safe to serve several conversations from.
 *
 * Greedy sampling throughout — a sampler with its own state would be a second
 * explanation for a difference, and this test is about the cache.
 */
class Main {
    static inline var STEPS = 12;

    /** Argmax over the final row. A prompt pass returns [seq, vocab]; only the
        last row predicts the next token. */
    static function greedy(logits:Tensor):Int {
        var shape = logits.shape();
        var vocab = shape[shape.length - 1];
        var rows = Std.int(logits.numel() / vocab);
        var base = (rows - 1) * vocab;
        var best = 0;
        var bv = logits.getFlat(base);
        for (i in 1...vocab) {
            var v = logits.getFlat(base + i);
            if (v > bv) { bv = v; best = i; }
        }
        return best;
    }

    static function step(model:LlamaModel, s:KVSession, ids:Array<Int>):Int {
        var logits = model.forwardIdsWith(ids, s);
        var id = greedy(logits);
        logits.free();
        return id;
    }

    static function solo(model:LlamaModel, promptIds:Array<Int>):Array<Int> {
        var s = model.newSession();
        var out:Array<Int> = [];
        var next = step(model, s, promptIds);
        out.push(next);
        for (_ in 1...STEPS) {
            next = step(model, s, [next]);
            out.push(next);
        }
        s.free();
        return out;
    }

    static function same(a:Array<Int>, b:Array<Int>):Bool {
        if (a.length != b.length) return false;
        for (i in 0...a.length) if (a[i] != b[i]) return false;
        return true;
    }

    static function show(label:String, ids:Array<Int>, tok:BPETokenizer):Void {
        Sys.println("[kv-session] " + label + " ids=" + ids.join(",")
            + "  text=" + StringTools.replace(tok.decode(ids), "\n", " "));
    }

    static function main():Void {
        var path = Sys.getEnvOr("GGUF", "");
        if (path == "") {
            Sys.println("error: set GGUF to a model file");
            Sys.exit(2);
        }
        var loader = new GGUFLoader();
        var loaded = loader.loadWithTokenizer(path, 1024);
        var tok = cast(loaded.tokenizer, BPETokenizer);
        var model = cast(loaded.model, LlamaModel);

        var promptA = tok.encode("The capital of France is");
        var promptB = tok.encode("Water boils at a temperature of");

        Sys.println("[kv-session] control: each conversation alone");
        var soloA = solo(model, promptA);
        var soloB = solo(model, promptB);
        show("A solo", soloA, tok);
        show("B solo", soloB, tok);

        Sys.println("[kv-session] test: both at once, one token each in turn");
        var sa = model.newSession();
        var sb = model.newSession();
        var interA:Array<Int> = [];
        var interB:Array<Int> = [];
        var nextA = step(model, sa, promptA);
        var nextB = step(model, sb, promptB);
        interA.push(nextA);
        interB.push(nextB);
        for (_ in 1...STEPS) {
            nextA = step(model, sa, [nextA]);
            nextB = step(model, sb, [nextB]);
            interA.push(nextA);
            interB.push(nextB);
        }
        show("A interleaved", interA, tok);
        show("B interleaved", interB, tok);

        var okA = same(soloA, interA);
        var okB = same(soloB, interB);
        Sys.println("[kv-session] lengths: A=" + sa.len() + " B=" + sb.len()
            + " (model's own cache untouched: " + model.cacheLen() + ")");
        sa.free();
        sb.free();

        if (okA && okB) {
            Sys.println("[kv-session] PASS: interleaving changed nothing");
            Sys.exit(0);
        }
        Sys.println("[kv-session] FAIL: A_match=" + okA + " B_match=" + okB);
        Sys.exit(1);
    }
}
