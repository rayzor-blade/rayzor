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
        var g = gate.forward(x.clone()).silu();
        var u = up.forward(x);
        return down.forward(g.mul(u));
    }

    public function parameters():Array<NamedTensor> {
        var ps = [];
        for (p in gate.parameters()) ps.push({ name: "ffn_gate." + p.name, tensor: p.tensor });
        for (p in up.parameters()) ps.push({ name: "ffn_up." + p.name, tensor: p.tensor });
        for (p in down.parameters()) ps.push({ name: "ffn_down." + p.name, tensor: p.tensor });
        return ps;
    }
}
