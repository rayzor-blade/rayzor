package nue;

import rayzor.ds.Tensor;

/**
 * Standard linear (matmul + optional bias) layer used by every transformer
 * projection (QKV, attention output, FFN gates).
 *
 * `weight` is `[out_features, in_features]` and the forward is
 * `y = x @ weight.T + bias`. We store weights in transposed form (as
 * `[in_features, out_features]`) so the matmul is a plain `x @ weight`
 * without a transpose step — matching the GGUF / llama.cpp convention.
 *
 * Bias is optional (Llama doesn't use bias on any linear).
 */
class Linear implements Module {
    public var weight:Tensor;
    public var bias:Null<Tensor>;
    public var paramName:String;

    public function new(weight:Tensor, ?bias:Tensor, paramName:String = "weight") {
        this.weight = weight;
        this.bias = bias;
        this.paramName = paramName;
    }

    public function forward(x:Tensor):Tensor {
        var y = x.matmul(weight);
        if (bias != null) y = y.add(bias);
        return y;
    }

    public function parameters():Array<NamedTensor> {
        var ps:Array<NamedTensor> = [{ name: paramName, tensor: weight }];
        if (bias != null) ps.push({ name: paramName + ".bias", tensor: bias });
        return ps;
    }
}
