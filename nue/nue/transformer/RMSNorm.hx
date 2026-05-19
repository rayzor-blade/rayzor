package nue.transformer;

import nue.Module;
import rayzor.ds.Tensor;

/**
 * RMS normalization with a learnable per-channel gain.
 *
 * Formula:
 * ```
 *   y = x / sqrt(mean(x^2) + eps) * weight
 * ```
 * applied independently to each row along the last dimension. `weight` is
 * a 1-D tensor sized to `head_dim` (or `hidden_dim` for the final norm).
 *
 * The actual sqrt-mean-square computation goes through the runtime kernel
 * `Tensor.rmsNorm`; this module owns the gain weight + composes them.
 *
 * Standard Llama config uses `eps = 1e-5`.
 */
class RMSNorm implements Module {
    public var weight:Tensor;
    public var eps:Float;
    public var paramName:String;

    public function new(weight:Tensor, eps:Float, paramName:String = "norm.weight") {
        this.weight = weight;
        this.eps = eps;
        this.paramName = paramName;
    }

    public function forward(x:Tensor):Tensor {
        return x.rmsNorm(eps).mul(weight);
    }

    public function parameters():Array<NamedTensor> {
        return [{ name: paramName, tensor: weight }];
    }
}
