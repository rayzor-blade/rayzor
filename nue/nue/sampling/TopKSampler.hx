package nue.sampling;

import rayzor.ds.Tensor;

/**
 * Top-K sampler. Restricts the candidate set to the K tokens with
 * the highest logits, then samples from a temperature-scaled
 * softmax over that subset.
 *
 * Keeps the model from picking truly improbable tokens (the long
 * tail), while preserving variety among the most plausible ones.
 * Useful default: K=40 with temperature ~0.7–0.9.
 *
 * Implementation: pre-allocates Float arrays + selection sort for K
 * rounds. The push-then-read-back pattern for `Array<Float>` is
 * broken in the current JIT, so all Float buffers are sized at
 * construction and assigned by index.
 */
class TopKSampler implements Sampler {
    public var k:Int;
    public var temperature:Float;
    private var state:Int;

    public function new(k:Int, temperature:Float, seed:Int) {
        this.k = k;
        this.temperature = temperature;
        this.state = seed;
    }

    public function sample(logits:Tensor):Int {
        var shape = logits.shape();
        var n = shape[shape.length - 1];
        var t = (temperature <= 0.0) ? 0.00000001 : temperature;
        var kk = (k > n) ? n : k;

        var logitsArr:Array<Float> = [for (_ in 0...n) 0.0];
        var ids:Array<Int> = [for (_ in 0...n) 0];
        for (i in 0...n) {
            logitsArr[i] = logits.getFlat(i);
            ids[i] = i;
        }
        // Selection sort for K rounds — moves the K largest to the front.
        for (round in 0...kk) {
            var bestIdx = round;
            for (j in (round + 1)...n) {
                if (logitsArr[j] > logitsArr[bestIdx]) bestIdx = j;
            }
            if (bestIdx != round) {
                var tmp = logitsArr[round];
                logitsArr[round] = logitsArr[bestIdx];
                logitsArr[bestIdx] = tmp;
                var ti = ids[round];
                ids[round] = ids[bestIdx];
                ids[bestIdx] = ti;
            }
        }

        var maxLogit = logitsArr[0];
        var probs:Array<Float> = [for (_ in 0...kk) 0.0];
        var total = 0.0;
        for (i in 0...kk) {
            var v = (logitsArr[i] - maxLogit) / t;
            var ev = Math.exp(v);
            probs[i] = ev;
            total += ev;
        }

        var r = nextFloat() * total;
        var acc = 0.0;
        for (i in 0...kk) {
            acc += probs[i];
            if (r <= acc) return ids[i];
        }
        return ids[kk - 1];
    }

    private function nextFloat():Float {
        state = (state * 1664525 + 1013904223) & 0x7FFFFFFF;
        return state / 2147483648.0;
    }
}
