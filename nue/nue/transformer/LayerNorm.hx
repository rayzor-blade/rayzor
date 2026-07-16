package nue.transformer;

import nue.Module;
import rayzor.ds.Tensor;

/**
 * Standard LayerNorm — `(x - mean) / sqrt(var + eps) * weight + bias`
 * over the last dimension.
 *
 * Used by GPT-2 / BERT / RoBERTa / DeBERTa and most pre-Llama
 * transformers. Llama-family models use `RMSNorm` instead (drops the
 * mean subtraction and the bias).
 *
 * The actual computation goes through the runtime kernel
 * `Tensor.layerNorm`; this module owns the gain + bias and composes
 * them.
 */
class LayerNorm implements Module {
    public var weight:Tensor;
    public var bias:Null<Tensor>;
    public var eps:Float;
    public var paramName:String;

    public function new(weight:Tensor, ?bias:Tensor, eps:Float = 0.00001, paramName:String) {
        this.weight = weight;
        this.bias = bias;
        this.eps = eps;
        this.paramName = paramName;
    }

    public function forward(x:Tensor):Tensor {
        // Bind the layerNorm result so it can be freed — as a chained temp
        // (`x.layerNorm(eps).mul(weight)`) it leaked one [seq, hidden] tensor
        // every call (~14 LayerNorms per BERT encode). mul() does not free its
        // receiver.
        var ln = x.layerNorm(eps);
        var y = ln.mul(weight);
        ln.free();
        // `y` is a fresh contiguous owning tensor straight out of `.mul`.
        // Accumulate bias in place — saves one [seq, hidden] F32 alloc on
        // every LayerNorm call. Cold for Llama (RMSNorm has no bias) but
        // warm for GPT-2 / BERT / RoBERTa families.
        if (bias != null) y.addInto(bias);
        return y;
    }

    public function parameters():Array<NamedTensor> {
        var ps:Array<NamedTensor> = [{ name: paramName, tensor: weight }];
        if (bias != null) ps.push({ name: paramName + ".bias", tensor: bias });
        return ps;
    }
}
