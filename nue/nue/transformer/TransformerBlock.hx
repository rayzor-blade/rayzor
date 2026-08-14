package nue.transformer;

import nue.Module;
import rayzor.ds.Tensor;

/**
 * Generic transformer block with two residual sub-layers, selectable
 * pre-norm or post-norm via the `postNorm` flag.
 *
 * ```
 *   pre-norm  (Llama/Mistral/Qwen, modern GPT):  h = x + sublayer(norm(x))
 *   post-norm (BERT/RoBERTa/original Transformer): h = norm(x + sublayer(x))
 * ```
 *
 * Every sub-layer is just a `Module`, so a single `TransformerBlock`
 * implementation drives many model families:
 *
 *   - **Llama / Mistral / Qwen** (pre-norm): `RMSNorm` + `GQAttention` + `RMSNorm` + `SwiGLU`
 *   - **BERT / RoBERTa** (post-norm): `LayerNorm` + `MultiHeadAttention` + `LayerNorm` + `GeluFFN`
 *
 * Norm placement is a genuine boolean that reorders the composition, not
 * a feature toggle that guts the class — so it is a flag rather than a
 * sibling type. BERT gets this wrong at its peril: the weights named
 * `attn_output_norm` / `layer_output_norm` are trained to normalize the
 * POST-residual sum; applying them pre-sublayer diverges every layer.
 *
 * The `name` field is used by `parameters()` to prefix the children's
 * tensor names — convention is `"blk.{layer}."` to match GGUF's naming.
 */
class TransformerBlock implements Module {
    public var attnNorm:Module;
    public var attn:Attention;
    public var ffnNorm:Module;
    public var ffn:Module;
    public var name:String;
    public var postNorm:Bool;

    public function new(
        attnNorm:Module, attn:Attention,
        ffnNorm:Module, ffn:Module,
        name:String, postNorm:Bool = false
    ) {
        this.attnNorm = attnNorm;
        this.attn = attn;
        this.ffnNorm = ffnNorm;
        this.ffn = ffn;
        this.name = name;
        this.postNorm = postNorm;
    }

    public function forward(x:Tensor):Tensor {
        return postNorm ? forwardPostNorm(x) : forwardPreNorm(x);
    }

    inline function forwardPreNorm(x:Tensor):Tensor {
        // Pre-norm residual: attnNorm consumes a copy of x; original x is
        // used again as the residual accumulator. `x.addInto(attnOut)`
        // mutates x's buffer in place (x += attnOut) — saves a fresh
        // [seq, hidden] F32 alloc per layer per token. x is a fresh
        // contiguous owning tensor here (gatherRows on layer 0, previous
        // .add result on layer N>0), so addInto's contiguity invariant
        // holds.
        //
        // Every named tensor below (clone, norm output, sublayer output)
        // is a fresh extern allocation. InsertFreePass doesn't recognise
        // tensor returns, so each intermediate gets an inline .free()
        // once its consumer is done — without these, every block leaks
        // six tensor handles per token (96 handles/token at 16 layers).
        var xClone1:Tensor = x.clone();
        var t:Tensor = attnNorm.forward(xClone1);
        xClone1.free();
        var attnOut:Tensor = attn.forward(t);
        t.free();
        x.addInto(attnOut);
        attnOut.free();
        var h1:Tensor = x;
        // Same pattern for the FFN sub-layer: ffnNorm consumes a copy of
        // h1, original h1 accumulates ffnOut in place.
        var xClone2:Tensor = h1.clone();
        var t2:Tensor = ffnNorm.forward(xClone2);
        xClone2.free();
        var ffnOut:Tensor = ffn.forward(t2);
        t2.free();
        h1.addInto(ffnOut);
        ffnOut.free();
        return h1;
    }

    inline function forwardPostNorm(x:Tensor):Tensor {
        // Post-norm residual (BERT): a = attnNorm(x + attn(x)), where the
        // sublayer consumes the RAW hidden state and the norm is applied
        // to the residual sum. Sublayer forwards never free their input
        // (the F32 matmul reads it), so x is cloned only to satisfy the
        // strict move analyzer's linearised call-arg consume; the clone
        // is freed right after, and the original x accumulates in place.
        var xc:Tensor = x.clone();
        var attnOut:Tensor = attn.forward(xc);
        xc.free();
        x.addInto(attnOut);            // x += attn(x)
        attnOut.free();
        var a:Tensor = attnNorm.forward(x);   // a = LayerNorm(x + attnOut)
        x.free();
        // FFN sublayer, same shape: out = ffnNorm(a + ffn(a)).
        var ac:Tensor = a.clone();
        var ffnOut:Tensor = ffn.forward(ac);
        ac.free();
        a.addInto(ffnOut);             // a += ffn(a)
        ffnOut.free();
        var out:Tensor = ffnNorm.forward(a);  // out = LayerNorm(a + ffnOut)
        a.free();
        return out;
    }

    /**
     * Post-norm block with an additive attention mask (BERT batched/padded
     * path). Identical to `forwardPostNorm` but the attention sublayer takes
     * an `attnBias`, reached through the `Attention` interface (virtual
     * dispatch — no downcast). Only the post-norm shape is masked; the sole
     * masked caller (BERT) is post-norm.
     */
    public function forwardMasked(x:Tensor, attnBias:Tensor):Tensor {
        var xc:Tensor = x.clone();
        var attnOut:Tensor = attn.forwardMasked(xc, attnBias); // virtual dispatch — no downcast
        xc.free();
        x.addInto(attnOut);            // x += attn(x, mask)
        attnOut.free();
        var a:Tensor = attnNorm.forward(x);   // a = LayerNorm(x + attnOut)
        x.free();
        var ac:Tensor = a.clone();
        var ffnOut:Tensor = ffn.forward(ac);
        ac.free();
        a.addInto(ffnOut);             // a += ffn(a)
        ffnOut.free();
        var out:Tensor = ffnNorm.forward(a);  // out = LayerNorm(a + ffnOut)
        a.free();
        return out;
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
