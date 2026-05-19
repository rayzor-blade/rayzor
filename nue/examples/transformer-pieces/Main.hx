import rayzor.ds.Tensor;
import rayzor.ds.DType;
import nue.Linear;
import nue.transformer.RMSNorm;
import nue.transformer.RoPE;
import nue.transformer.SwiGLU;
import nue.transformer.KVCache;
import nue.transformer.GQAttention;

class Main {
    static function main() {
        trace("=== nue transformer pieces ===");

        // --- RMSNorm ---
        var weight = Tensor.ones([4], F32);
        var norm = new RMSNorm(weight, 0.00001, "norm.weight");
        var x = Tensor.fromArray([1.0, 2.0, 3.0, 4.0], F32);
        var nx = norm.forward(x);
        trace("rmsnorm[0] ~ 0.365: " + nx.get([0]));
        trace("rmsnorm[3] ~ 1.460: " + nx.get([3]));

        // --- RoPE ---
        var rope = new RoPE(4, 8, 10000.0);
        trace("rope.cos[0,0] = " + rope.cos.get([0, 0]));

        // --- Linear ---
        var idW = Tensor.fromArray([1.0, 0.0, 0.0, 1.0], F32).reshape([2, 2]);
        var lin = new Linear(idW, null, "weight");
        var ly = lin.forward(Tensor.fromArray([3.0, 7.0], F32).reshape([1, 2]));
        trace("identity-linear[0,0] = " + ly.get([0, 0]));
        trace("identity-linear[0,1] = " + ly.get([0, 1]));

        // --- SwiGLU ---
        var swi = new SwiGLU(new Linear(idW, null, "gate"), new Linear(idW, null, "up"), new Linear(idW, null, "down"));
        var swy = swi.forward(Tensor.fromArray([2.0, 3.0], F32).reshape([1, 2]));
        trace("swiglu[0,0] ~ 3.52: " + swy.get([0, 0]));
        trace("swiglu[0,1] ~ 8.57: " + swy.get([0, 1]));

        // --- New Tensor primitives ---

        // 3-D batched matmul: 2 batches of [2,2] × [2,2]
        var ba = Tensor.fromArray([
            // batch 0: [[1,2],[3,4]]
            1.0, 2.0, 3.0, 4.0,
            // batch 1: [[5,6],[7,8]]
            5.0, 6.0, 7.0, 8.0
        ], F32).reshape([2, 2, 2]);
        var bb = Tensor.fromArray([
            // batch 0: [[1,0],[0,1]] (identity)
            1.0, 0.0, 0.0, 1.0,
            // batch 1: [[2,0],[0,2]] (2x scale)
            2.0, 0.0, 0.0, 2.0
        ], F32).reshape([2, 2, 2]);
        var bc = ba.bmm(bb);
        trace("bmm[0,0,0] ~ 1: " + bc.get([0, 0, 0]));   // identity → 1
        trace("bmm[0,1,1] ~ 4: " + bc.get([0, 1, 1]));   // identity → 4
        trace("bmm[1,0,0] ~ 10: " + bc.get([1, 0, 0])); // 5 * 2 = 10
        trace("bmm[1,1,1] ~ 16: " + bc.get([1, 1, 1])); // 8 * 2 = 16

        // scale
        var sc = Tensor.fromArray([1.0, 2.0, 3.0, 4.0], F32).scale(0.5);
        trace("scale[0] ~ 0.5: " + sc.get([0]));
        trace("scale[3] ~ 2.0: " + sc.get([3]));

        // transposeLast2
        var m = Tensor.fromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], F32).reshape([2, 3]);
        var mt = m.transposeLast2();
        trace("transpose[0,0] = " + mt.get([0, 0])); // 1
        trace("transpose[1,0] = " + mt.get([1, 0])); // 2
        trace("transpose[2,1] = " + mt.get([2, 1])); // 6

        // causalMask_ in place. Build a 1x3x3 zeros tensor, mask, inspect.
        var mask = Tensor.zeros([1, 3, 3], F32);
        mask.causalMask_(0);
        // Row 0: cols 1,2 → -inf; col 0 → 0.
        // Row 1: col 2 → -inf; cols 0,1 → 0.
        // Row 2: no -inf.
        trace("mask[0,0,0] = 0: " + mask.get([0, 0, 0]));
        trace("mask[0,1,2] = -inf: " + mask.get([0, 1, 2]));
        trace("mask[0,2,2] = 0: " + mask.get([0, 2, 2]));

        // --- GQAttention end-to-end (tiny synthetic config) ---
        // hidden_size = 4, num_q_heads = 2, num_kv_heads = 1, head_dim = 2.
        // Group factor G = 2 (both Q heads share the one KV head).
        var hidden = 4;
        var numQH = 2;
        var numKVH = 1;
        var hd = 2;

        // Identity projections so we can reason about the math.
        var I4 = Tensor.fromArray([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ], F32).reshape([4, 4]);
        var qP = new Linear(I4, null, "weight");
        // K and V project to hidden=2 (numKVH * hd). Use the first 2 cols
        // of identity → drops the last 2 input features.
        var I4to2 = Tensor.fromArray([
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 0.0
        ], F32).reshape([4, 2]);
        var kP = new Linear(I4to2, null, "weight");
        var vP = new Linear(I4to2, null, "weight");
        var oP = new Linear(I4, null, "weight"); // hidden=4 → 4

        var attnRope = new RoPE(hd, 16, 10000.0);
        var cache = new KVCache(16, numKVH, hd, F32);
        var attn = new GQAttention(qP, kP, vP, oP, attnRope, cache, numQH, numKVH, hd);

        // Single-token input.
        var xin = Tensor.fromArray([1.0, 2.0, 3.0, 4.0], F32).reshape([1, hidden]);
        var y = attn.forward(xin);
        trace("attn out shape = " + y.shape()[0] + "," + y.shape()[1]); // 1,4
        // Single-token self-attention with full causal mask is just the
        // (rotated, scaled) projection of itself into V — so the output
        // shape and the parameter wiring is what we sanity-check here;
        // exact values depend on RoPE phase + softmax(self-score)=1.
        trace("attn out[0,0] = " + y.get([0, 0]));
        trace("attn out[0,3] = " + y.get([0, 3]));
        trace("cache currentLen after one token = " + attn.cache.currentLen); // 1

        // Append a second token, verify cache grows + decode path runs.
        var xin2 = Tensor.fromArray([0.5, 0.5, 0.5, 0.5], F32).reshape([1, hidden]);
        var y2 = attn.forward(xin2);
        trace("attn out2 shape = " + y2.shape()[0] + "," + y2.shape()[1]);
        trace("cache currentLen after two tokens = " + attn.cache.currentLen); // 2

        // Parameter enumeration
        var ps = attn.parameters();
        trace("GQAttention params count = " + ps.length); // 4: q, k, v, o

        trace("=== Done ===");
    }
}
