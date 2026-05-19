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
        var y = x.layerNorm(eps).mul(weight);
        if (bias != null) y = y.add(bias);
        return y;
    }

    public function parameters():Array<NamedTensor> {
        var ps:Array<NamedTensor> = [{ name: paramName, tensor: weight }];
        if (bias != null) ps.push({ name: paramName + ".bias", tensor: bias });
        return ps;
    }
}
