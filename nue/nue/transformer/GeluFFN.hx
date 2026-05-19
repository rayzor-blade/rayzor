package nue.transformer;

import nue.Module;
import nue.Linear;
import rayzor.ds.Tensor;

/**
 * Classical two-layer FFN with GELU activation — the original
 * Transformer / GPT-2 / BERT feed-forward block.
 *
 * ```
 *   h = gelu(x @ W_up + b_up)
 *   y = h @ W_down + b_down
 * ```
 *
 * The Llama-family `SwiGLU` block is the gated alternative; both are
 * drop-in replacements at the `Module` level so a `TransformerBlock`
 * doesn't care which one its `ffn` field is.
 */
class GeluFFN implements Module {
    public var up:Linear;
    public var down:Linear;

    public function new(up:Linear, down:Linear) {
        this.up = up;
        this.down = down;
    }

    public function forward(x:Tensor):Tensor {
        return down.forward(up.forward(x).gelu());
    }

    public function parameters():Array<NamedTensor> {
        var ps = [];
        for (p in up.parameters()) ps.push({ name: "ffn_up." + p.name, tensor: p.tensor });
        for (p in down.parameters()) ps.push({ name: "ffn_down." + p.name, tensor: p.tensor });
        return ps;
    }
}
