import nue.sampling.Sampler;
import nue.sampling.ArgmaxSampler;
import nue.sampling.TemperatureSampler;
import nue.sampling.TopKSampler;
import nue.sampling.TopPSampler;
import rayzor.ds.Tensor;
import rayzor.ds.DType;

class Main {
    static function main() {
        trace("=== Phase 8: sampling surface check ===");

        var logits = Tensor.fromArray([0.1, 0.2, 0.3, 5.0, 0.5], F32);

        var greedy = new ArgmaxSampler();
        trace("argmax:");
        for (i in 0...3) trace("  " + greedy.sample(logits));

        var coldTemp = new TemperatureSampler(0.0001, 42);
        trace("temperature=1e-4 (should be 3):");
        for (i in 0...3) trace("  " + coldTemp.sample(logits));

        var warmTemp = new TemperatureSampler(2.0, 7);
        trace("temperature=2.0 (valid range 0..4):");
        for (i in 0...5) {
            var t = warmTemp.sample(logits);
            trace("  " + t + " inRange=" + (t >= 0 && t < 5));
        }

        var tk = new TopKSampler(2, 0.8, 123);
        trace("top-k=2 (should be 3 or 4):");
        for (i in 0...5) {
            var t = tk.sample(logits);
            trace("  " + t + " topTwo=" + (t == 3 || t == 4));
        }

        var tp = new TopPSampler(0.5, 0.8, 999);
        trace("top-p=0.5 (should mostly be 3):");
        for (i in 0...5) trace("  " + tp.sample(logits));

        trace("=== Done ===");
    }
}
