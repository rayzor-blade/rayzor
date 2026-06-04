package nue.sampling;

import rayzor.ds.Tensor;

/**
 * Greedy decoder — picks the token with the highest logit at each
 * step. Deterministic; same prompt always produces the same output.
 *
 * Useful as a baseline (sanity-check the model wiring works) and for
 * tasks where you want repeatable behaviour (eval suites, code
 * completion, structured output extraction).
 *
 * Quality drops sharply on creative tasks — without any
 * stochasticity, the model gets stuck in repetitive loops on
 * open-ended generation. Use a temperature/top-p sampler there
 * instead.
 */
class ArgmaxSampler implements Sampler {
    public function new() {}

    public function sample(logits:Tensor):Int {
        var shape = logits.shape();
        var n = shape[shape.length - 1];
        var bestId = 0;
        var bestLogit = logits.getFlat(0);
        for (i in 1...n) {
            var v = logits.getFlat(i);
            if (v > bestLogit) {
                bestLogit = v;
                bestId = i;
            }
        }
        return bestId;
    }
}
