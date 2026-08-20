package nue.transformer;

import nue.Module;
import nue.Linear;
import rayzor.ds.Tensor;

/**
 * SwiGLU feed-forward block from the Llama family.
 *
 * Three linear projections, no new kernels:
 * ```
 *   gate = x @ W_gate
 *   up   = x @ W_up
 *   y    = (silu(gate) * up) @ W_down
 * ```
 *
 * `silu(x) = x * sigmoid(x)` — provided by `Tensor.silu()` (existing).
 * `*` is elementwise multiplication. The `down` projection brings the
 * hidden dim back to `hidden_size`. Standard Llama FFN dim is
 * `~2.66 * hidden_size`, rounded to a multiple of 256.
 */
class SwiGLU implements Module {
    public var gate:Linear;
    public var up:Linear;
    public var down:Linear;

    // ---- planned route ------------------------------------------------
    // Which route this module takes, decided once when the model is built
    // instead of re-derived from environment flags on every forward.
    //
    // Zero means unplanned, and unplanned means "work it out live, exactly as
    // before" — a module built outside the planner (the standalone examples do
    // this) keeps today's behaviour without edit. Appended rather than
    // inserted: an importing module resolves fields by declaration order.
    //   0 unplanned   1 on   2 off   3 on and verify   4 off and verify
    public var planHaxeMat:Int = 0;
    public var planFusedPair:Int = 0;

    public function new(gate:Linear, up:Linear, down:Linear) {
        this.gate = gate;
        this.up = up;
        this.down = down;
    }

    public function forward(x:Tensor):Tensor {
        // SwiGLU shares the same input x across gate and up projections.
        // The down projection then takes the elementwise product. Two
        // independent consumers of x → clone the first one.
        //
        // Manual frees: extern tensor allocs are runtime-managed via
        // the tensor pool's ARC, but the JIT's InsertFreePass doesn't
        // know about rayzor_tensor_free, so every intermediate here
        // leaks unless we release it inline. Five frees per FFN call
        // × 16 layers per token compounds fast on long generations.
        // gate and up read the SAME x: when both are quantized, run them
        // through the fused kernel — one Q8_K quantize + one pool dispatch
        // over the concatenated row space instead of two of each
        // (bit-identical per row to the two separate calls).
        var gLin:Tensor;
        var u:Tensor;
        var gwq = gate.qweight;
        var uwq = up.qweight;
        // gate/up share the post-FFN-norm activation, so a row-wise (INT8/Q8_0)
        // pair fuses by default — one activation quantise instead of two.
        // Separate, cheaper mechanism than the opt-in k-quant fusion.
        // The route this layer was built for; zero means decide it here, as
        // before. Kept separate from attention's: this site treats a disabled
        // Haxe matmul as "split", where attention treats it as "fused".
        var useHaxeMat = planHaxeMat != 0
            ? planHaxeMat == 1
            : Linear.useHaxeMatmul();
        var useFusedPair = planFusedPair != 0
            ? planFusedPair == 1
            : (nue.Q4Matmul.useFusedMatmul() || nue.Q4Matmul.canFuseRowwise(gwq, uwq, null));
        nue.Q4Matmul.noteFusionSite(false, gwq != null && uwq != null, useHaxeMat);
        if (useHaxeMat && gwq != null && uwq != null && useFusedPair) {
            var pair = nue.Q4Matmul.matmulFused(gwq, uwq, null, x, gate.pool);
            gLin = pair[0];
            u = pair[1];
        } else {
            var xClone = x.clone();
            gLin = gate.forward(xClone);
            xClone.free();
            u = up.forward(x);
        }
        var gu = gLin.siluMul(u);
        gLin.free();
        u.free();
        var result = down.forward(gu);
        gu.free();
        return result;
    }

    public function parameters():Array<NamedTensor> {
        var ps = [];
        for (p in gate.parameters()) ps.push({ name: "ffn_gate." + p.name, tensor: p.tensor });
        for (p in up.parameters()) ps.push({ name: "ffn_up." + p.name, tensor: p.tensor });
        for (p in down.parameters()) ps.push({ name: "ffn_down." + p.name, tensor: p.tensor });
        return ps;
    }
}
