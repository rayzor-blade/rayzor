package nue.sampling;

import rayzor.ds.Tensor;

/**
 * Token-sampling strategy. Consumes the model's per-step logits
 * (shape `[vocab_size]`, one F32 per token) and returns a single
 * token ID to feed back into the next decode step.
 *
 * Samplers are stateless across calls — they take a fresh logits
 * tensor each step and return a fresh ID. Stateful sampling
 * (e.g. repetition penalty over a sliding window of recent tokens)
 * lives in a wrapping sampler that owns the window.
 *
 * Implementations are expected to be:
 *   - **CPU-side.** Logits are typically small enough (32k–128k) that
 *     a CPU pass is fine; chasing the last 100µs by moving sampling
 *     to GPU isn't worth the dispatch overhead for now.
 *   - **Pure functions.** Same input logits → same output (for
 *     deterministic ones) or same probability distribution (for
 *     stochastic ones with a fixed RNG seed). No hidden side effects.
 */
interface Sampler {
    /** Pick the next token ID given the model's output logits. */
    function sample(logits:Tensor):Int;
}
