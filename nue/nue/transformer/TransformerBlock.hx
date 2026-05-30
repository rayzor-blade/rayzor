package nue.transformer;

import nue.Module;
import rayzor.ds.Tensor;

/**
 * Generic pre-norm transformer block with two residual sub-layers.
 *
 * ```
 *   h1 = x + attn(attnNorm(x))
 *   h2 = h1 + ffn(ffnNorm(h1))
 *   return h2
 * ```
 *
 * Every sub-layer is just a `Module`, so a single `TransformerBlock`
 * implementation drives many model families:
 *
 *   - **Llama / Mistral / Qwen**: `RMSNorm` + `GQAttention` + `RMSNorm` + `SwiGLU`
 *   - **GPT-2 / OPT**: `LayerNorm` + `MultiHeadAttention` + `LayerNorm` + `GeluFFN`
 *   - **Falcon**: parallel attn + FFN (override `forward` for that variant)
 *
 * Pre-norm vs post-norm: this implementation is pre-norm (norm before
 * sublayer, residual added afterwards) — the convention every modern
 * open-weight model uses. Subclass and override `forward` if you need
 * the original GPT-style post-norm.
 *
 * The `name` field is used by `parameters()` to prefix the children's
 * tensor names — convention is `"blk.{layer}."` to match GGUF's naming.
 */
class TransformerBlock implements Module {
    public var attnNorm:Module;
    public var attn:Module;
    public var ffnNorm:Module;
    public var ffn:Module;
    public var name:String;

    public function new(
        attnNorm:Module, attn:Module,
        ffnNorm:Module, ffn:Module,
        name:String
    ) {
        this.attnNorm = attnNorm;
        this.attn = attn;
        this.ffnNorm = ffnNorm;
        this.ffn = ffn;
        this.name = name;
    }

    public function forward(x:Tensor):Tensor {
        // Pre-norm residual: attnNorm consumes a copy of x; original x is
        // used again as the residual accumulator. `x.addInto(attnOut)`
        // mutates x's buffer in place (x += attnOut) — saves a fresh
        // [seq, hidden] F32 alloc per layer per token. x is a fresh
        // contiguous owning tensor here (gatherRows on layer 0, previous
        // .add result on layer N>0), so addInto's contiguity invariant
        // holds.
        var t = attnNorm.forward(x.clone());
        var attnOut = attn.forward(t);
        x.addInto(attnOut);
        var h1 = x;
        // Same pattern for the FFN sub-layer: ffnNorm consumes a copy of
        // h1, original h1 accumulates ffnOut in place.
        var t2 = ffnNorm.forward(h1.clone());
        var ffnOut = ffn.forward(t2);
        h1.addInto(ffnOut);
        return h1;
    }

    public function parameters():Array<NamedTensor> {
        var ps = [];
        for (p in attnNorm.parameters()) ps.push({ name: name + "attn_norm." + p.name, tensor: p.tensor });
        for (p in attn.parameters()) ps.push({ name: name + p.name, tensor: p.tensor });
        for (p in ffnNorm.parameters()) ps.push({ name: name + "ffn_norm." + p.name, tensor: p.tensor });
        for (p in ffn.parameters()) ps.push({ name: name + p.name, tensor: p.tensor });
        return ps;
    }
}
