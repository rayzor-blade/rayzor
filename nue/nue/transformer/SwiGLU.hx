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
        // Pure-Haxe kernel path: gate and up read the SAME x, so run them
        // through the fused kernel — one Q8_K quantize + one pool dispatch
        // over the concatenated row space instead of two of each.
        // Bit-identical per row to the two separate calls.
        var gLin:Tensor;
        var u:Tensor;
        var gwq = gate.qweight;
        var uwq = up.qweight;
        if (Linear.useHaxeMatmul() && gwq != null && uwq != null) {
            var pair = nue.Q4Matmul.matmulFused(gwq, uwq, null, x, gate.pool);
            gLin = pair[0];
            u = pair[1];
        } else {
            var xClone = x.clone();
            gLin = gate.forward(xClone);
            xClone.free();
            u = up.forward(x);
        }
        var g = gLin.silu();
        gLin.free();
        var gu = g.mul(u);
        g.free();
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
