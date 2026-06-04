package nue.sampling;

import rayzor.ds.Tensor;

/**
 * Temperature-scaled softmax sampler. Divides every logit by
 * `temperature` before the softmax, then samples a token
 * proportional to the resulting probability mass.
 *
 * - `temperature → 0`  collapses to argmax (greedy).
 * - `temperature = 1`  matches the model's native distribution.
 * - `temperature > 1`  flattens the distribution (more random).
 *
 * Typical values for chat-style generation: 0.7–0.9. For factual /
 * deterministic tasks, anneal towards 0 or just use `ArgmaxSampler`.
 *
 * **RNG.** Linear-congruential generator on a 31-bit state. Seed
 * once at construction; subsequent calls advance the state, so the
 * same seed reproduces the same sequence.
 *
 * **Why pre-allocate `probs` instead of `push`:** the current JIT
 * has a bug where `Array<Float>.push(f)` stores the value but reads
 * back garbage from `arr[i]`. Pre-allocating via comprehension and
 * assigning by index sidesteps it. Same workaround is applied to
 * TopK / TopP samplers.
 */
class TemperatureSampler implements Sampler {
    public var temperature:Float;
    private var state:Int;

    public function new(temperature:Float, seed:Int) {
        this.temperature = temperature;
        this.state = seed;
    }

    public function sample(logits:Tensor):Int {
        var shape = logits.shape();
        var n = shape[shape.length - 1];

        var t = (temperature <= 0.0) ? 0.00000001 : temperature;
        var maxLogit = logits.getFlat(0);
        for (i in 1...n) {
            var v = logits.getFlat(i);
            if (v > maxLogit) maxLogit = v;
        }

        var probs:Array<Float> = [for (_ in 0...n) 0.0];
        var total = 0.0;
        for (i in 0...n) {
            var v = (logits.getFlat(i) - maxLogit) / t;
            var ev = Math.exp(v);
            probs[i] = ev;
            total += ev;
        }

        var r = nextFloat() * total;
        var acc = 0.0;
        for (i in 0...n) {
            acc += probs[i];
            if (r <= acc) return i;
        }
        return n - 1;
    }

    private function nextFloat():Float {
        state = (state * 1664525 + 1013904223) & 0x7FFFFFFF;
        return state / 2147483648.0;
    }
}
