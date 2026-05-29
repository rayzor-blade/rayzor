package nue;

import rayzor.ds.Tensor;
import rayzor.ds.QTensor;
import rayzor.ds.DType;

/**
 * Standard linear (matmul + optional bias) layer used by every transformer
 * projection (QKV, attention output, FFN gates).
 *
 * `weight` has the PyTorch shape `[out_features, in_features]` and the
 * forward is `y = x @ weight.T + bias`. The transpose is performed at
 * the kernel level by `Tensor.matmulT`, which iterates as a row-against-
 * row dot product (SIMD-friendly) and avoids the strided column access
 * a literal `weight.T` view would force the F32 matmul into.
 *
 * This layout matches both PyTorch's `nn.Linear.weight` and GGUF's
 * on-disk tensor orientation (PyTorch row-major `[out, in]`), so loaded
 * weights drop in without per-tensor transpose passes.
 *
 * Phase 4b: when constructed via `Linear.fromQuant`, the layer holds a
 * `QTensor` weight (Q4_K_M, also `[out, in]`) and forwards through the
 * fused dequant-matmul kernel `QTensor.matmulXTQ(x)` — no F32 copy of
 * the weight is ever materialised. Memory drops ~8× for Q4_K_M weights
 * vs the dequant path. In that mode the F32 `weight` slot holds a tiny
 * 1×1 sentinel (kept non-null deliberately — a `Null<Tensor>` field
 * shifts the class layout and surfaces a latent JIT bug in large
 * import graphs); `parameters()` returns an empty list since the
 * quantised storage isn't an F32 `Tensor`.
 *
 * Bias is optional (Llama doesn't use bias on any linear; Qwen2 does on
 * Q/K/V projections).
 */
class Linear implements Module {
    public var weight:Tensor;
    public var qweight:Null<QTensor>;
    public var bias:Null<Tensor>;
    public var paramName:String;

    public function new(weight:Tensor, ?bias:Tensor, paramName:String = "weight") {
        this.weight = weight;
        this.qweight = null;
        this.bias = bias;
        this.paramName = paramName;
    }

    /**
     * Build a Linear whose weight is a `QTensor` (compressed Q4_K_M
     * storage). Forward runs the fused dequant-matmul kernel — no F32
     * weight is allocated. The F32 `weight` slot is filled with a 1×1
     * sentinel to keep the class layout stable (see class doc).
     */
    public static function fromQuant(qweight:QTensor, ?bias:Tensor, paramName:String = "weight"):Linear {
        // Bare `F32` — TAST enum-variant disambiguation picks DType.F32 because
        // Tensor.zeros's dtype param is typed as DType. Before the disambiguation
        // landed, scope-walk could land on MetaValue.F32 (different enum, same
        // simple name) and silently pass a pointer to dtype. See
        // bugs_dtype_enum_cross_file_pointer.
        var sentinel = Tensor.zeros([1, 1], F32);
        var l = new Linear(sentinel, bias, paramName);
        l.qweight = qweight;
        return l;
    }

    public function forward(x:Tensor):Tensor {
        var y:Tensor;
        if (qweight != null) {
            y = qweight.matmulXTQ(x);
        } else {
            y = x.matmulT(weight);
        }
        if (bias != null) y = y.add(bias);
        return y;
    }

    public function parameters():Array<NamedTensor> {
        if (qweight != null) {
            // Quantised — `weight` is a 1×1 sentinel, not a real parameter.
            var ps:Array<NamedTensor> = [];
            if (bias != null) ps.push({ name: paramName + ".bias", tensor: bias });
            return ps;
        }
        var ps:Array<NamedTensor> = [{ name: paramName, tensor: weight }];
        if (bias != null) ps.push({ name: paramName + ".bias", tensor: bias });
        return ps;
    }
}
