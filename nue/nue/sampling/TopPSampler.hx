package nue.sampling;

import rayzor.ds.Tensor;

/**
 * Top-P / nucleus sampler. Sorts tokens by probability, then keeps
 * the smallest set whose cumulative mass first exceeds `p`. Samples
 * from that nucleus with a temperature-scaled softmax.
 *
 * Adapts the candidate set size to the model's confidence: when the
 * model has a clear winner, only a few tokens make the cut; when
 * the distribution is flat, many do. Common combination: `p = 0.9`,
 * `temperature = 0.7`.
 *
 * All Float buffers are pre-allocated and assigned by index to
 * sidestep the JIT's broken `Array<Float>.push` read-back path.
 */
class TopPSampler implements Sampler {
    public var p:Float;
    public var temperature:Float;
    private var state:Int;

    public function new(p:Float, temperature:Float, seed:Int) {
        this.p = p;
        this.temperature = temperature;
        this.state = seed;
    }

    public function sample(logits:Tensor):Int {
        var shape = logits.shape();
        var n = shape[shape.length - 1];
        var t = (temperature <= 0.0) ? 0.00000001 : temperature;

        var probs:Array<Float> = [for (_ in 0...n) 0.0];
        var ids:Array<Int> = [for (_ in 0...n) 0];
        var maxLogit = logits.getFlat(0);
        for (i in 1...n) {
            var v = logits.getFlat(i);
            if (v > maxLogit) maxLogit = v;
        }
        var total = 0.0;
        for (i in 0...n) {
            var v = (logits.getFlat(i) - maxLogit) / t;
            var ev = Math.exp(v);
            probs[i] = ev;
            ids[i] = i;
            total += ev;
        }
        for (i in 0...n) probs[i] = probs[i] / total;

        // Insertion sort descending by probability.
        for (i in 1...n) {
            var pv = probs[i];
            var iv = ids[i];
            var j = i;
            while (j > 0 && probs[j - 1] < pv) {
                probs[j] = probs[j - 1];
                ids[j] = ids[j - 1];
                j--;
            }
            probs[j] = pv;
            ids[j] = iv;
        }

        var cutoff = n;
        var acc = 0.0;
        for (i in 0...n) {
            acc += probs[i];
            if (acc >= p) { cutoff = i + 1; break; }
        }

        var kept = 0.0;
        for (i in 0...cutoff) kept += probs[i];
        var r = nextFloat() * kept;
        var run = 0.0;
        for (i in 0...cutoff) {
            run += probs[i];
            if (r <= run) return ids[i];
        }
        return ids[cutoff - 1];
    }

    private function nextFloat():Float {
        state = (state * 1664525 + 1013904223) & 0x7FFFFFFF;
        return state / 2147483648.0;
    }
}
