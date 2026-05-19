import rayzor.ds.Tensor;
import rayzor.ds.DType;
import nue.Linear;
import nue.transformer.RMSNorm;
import nue.transformer.RoPE;
import nue.transformer.SwiGLU;
import nue.transformer.KVCache;

class Main {
    static function main() {
        trace("=== nue transformer pieces ===");

        // --- RMSNorm ---
        var weight = Tensor.ones([4], F32);
        var norm = new RMSNorm(weight, 0.00001, "norm.weight");
        var x = Tensor.fromArray([1.0, 2.0, 3.0, 4.0], F32);
        var nx = norm.forward(x);
        // x / sqrt(mean(x^2)) ≈ x / 2.7386 — sum-of-squares = 30 → mean = 7.5 → sqrt = 2.7386
        trace("rmsnorm[0] ~ 0.365: " + nx.get([0]));
        trace("rmsnorm[3] ~ 1.460: " + nx.get([3]));

        // --- RoPE ---
        var rope = new RoPE(4, 8, 10000.0);
        trace("rope.cos[0,0] = " + rope.cos.get([0, 0]));  // 1.0

        var q = Tensor.fromArray([1.0, 0.0, 1.0, 0.0], F32).reshape([1, 1, 4]);
        var qr = rope.apply(q, 0);
        trace("rope identity[0,0,0]: " + qr.get([0, 0, 0]));  // 1.0

        // --- Linear ---
        var lw = Tensor.fromArray([
            1.0, 0.0,
            0.0, 1.0
        ], F32).reshape([2, 2]);
        var lin = new Linear(lw);
        var ly = lin.forward(Tensor.fromArray([3.0, 7.0], F32).reshape([1, 2]));
        trace("identity-linear[0,0] = " + ly.get([0, 0]));  // 3.0
        trace("identity-linear[0,1] = " + ly.get([0, 1]));  // 7.0

        // --- SwiGLU (very small dims, identity-ish weights) ---
        var idW = Tensor.fromArray([
            1.0, 0.0,
            0.0, 1.0
        ], F32).reshape([2, 2]);
        var swi = new SwiGLU(
            new Linear(idW, null, "gate"),
            new Linear(idW, null, "up"),
            new Linear(idW, null, "down")
        );
        var swx = Tensor.fromArray([2.0, 3.0], F32).reshape([1, 2]);
        var swy = swi.forward(swx);
        // silu(2)*2 = 1.762*2 = 3.523; silu(3)*3 = 2.857*3 = 8.572
        trace("swiglu[0,0] ~ 3.52: " + swy.get([0, 0]));
        trace("swiglu[0,1] ~ 8.57: " + swy.get([0, 1]));

        // --- KVCache ---
        var cache = new KVCache(8, 2, 4, F32);
        trace("cache.currentLen = " + cache.currentLen);  // 0

        var newK = Tensor.fromArray([
            1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0
        ], F32).reshape([1, 2, 4]);
        var newV = newK;  // alias
        cache.append(newK, newV);
        trace("after append currentLen = " + cache.currentLen);  // 1

        var ks = cache.keysView();
        trace("keysView shape = " + ks.shape()[0] + "," + ks.shape()[1] + "," + ks.shape()[2]);  // 1,2,4
        trace("keysView[0,0,0] = " + ks.get([0, 0, 0]));  // 1.0
        trace("keysView[0,1,3] = " + ks.get([0, 1, 3]));  // 8.0

        // --- Parameter enumeration ---
        var ps = norm.parameters();
        trace("RMSNorm params count = " + ps.length);  // 1
        trace("RMSNorm param[0].name = " + ps[0].name);

        var swiParams = swi.parameters();
        trace("SwiGLU params count = " + swiParams.length);  // 3

        trace("=== Done ===");
    }
}
