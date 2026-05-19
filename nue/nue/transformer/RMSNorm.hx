package nue.transformer;

import nue.Module;
import rayzor.ds.Tensor;

/**
 * RMS normalization with a learnable per-channel gain.
 *
 * Formula:
 * ```
 *   y = x / sqrt(mean(x²) + eps) * weight
 * ```
 * applied independently to each row along the last dimension. `weight` is
 * a 1-D tensor sized to `head_dim` (or `hidden_dim` for the final norm).
 *
 * The actual sqrt-mean-square computation goes through the runtime kernel
 * `Tensor.rmsNorm`; this module owns the gain weight + composes them.
 *
 * Standard Llama config uses `eps = 1e-5`.
 *
 * **GPU dispatch.** This module is CPU-only by design. For on-device
 * inference, call `rayzor.gpu.GPUCompute.rmsNorm(xGpu, weightGpu, rowLen, eps)`
 * directly with weights pre-uploaded via `gpu.createBuffer(weight)`. The
 * "wire it through the module API" path tripped JIT-compiler quirks
 * around mixed-typed fields on `nue.Module` implementers (see
 * `bugs_known.md`); revisit when those are fixed.
 */
class RMSNorm implements Module {
    public var weight:Tensor;
    public var eps:Float;
    public var paramName:String;

    public function new(weight:Tensor, eps:Float, paramName:String) {
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
