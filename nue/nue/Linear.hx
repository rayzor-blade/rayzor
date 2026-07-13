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
    /** Persistent spin pool for the pure-Haxe quant matmul; set by the
        arch builder. Plumbed as an instance — cross-module object statics
        read garbage (statics aren't forwarded across modules). */
    public var pool:Null<rayzor.concurrent.SpinPool> = null;

    public function new(weight:Tensor, ?bias:Tensor, paramName:String = "weight") {
        this.weight = weight;
        this.qweight = null;
        this.bias = bias;
        this.paramName = paramName;
    }

    /** RAYZOR_HAXE_MATMUL=1 routes quantised forwards through
        `Q4Matmul.matmul`; unset/0 uses the runtime kernel. Cached after
        the first read. */
    // 0 = uninitialised, 1 = on, 2 = off. Zero-valued "uninitialised" is
    // load-bearing: a cross-module duplicate of this static starts at 0
    // (field initialisers don't run for duplicated statics), and with a -1
    // sentinel such a copy would return false forever instead of re-reading
    // the env.
    static var _haxeMatmul:Int = 0;

    public static function useHaxeMatmul():Bool {
        if (_haxeMatmul == 0) {
            var v = Sys.getEnvOr("NUE_MATMUL", "RAYZOR_HAXE_MATMUL");
            _haxeMatmul = (v != null && v != "0" && v != "" && v != "false") ? 1 : 2;
        }
        return _haxeMatmul == 1;
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
            // Multi-thread the per-output-row Q4_K_M / Q6_K dequant +
            // dot via a runtime-side fork-join (`std::thread::scope`).
            // `0` picks the auto default (6 workers on M1 Pro). On 1B
            // Q4_K_M this is the entire CPU hot loop — threading it is
            // the single biggest end-to-end win available without a
            // custom SIMD kernel.
            //
            // `x.clone()` here because the strict E0382 analyzer
            // linearises the if/else and would otherwise see this
            // branch's consume followed by the else-branch's
            // `x.matmulT(weight)` as a use-after-move. Cloning the
            // unused path is cheaper than rebinding through a
            // mutable wrapper.
            //
            // Bind the clone to a local so we can drop the bumped
            // refcount explicitly after the kernel consumes it. The
            // compiler's InsertFreePass doesn't recognise tensor
            // returns (extern allocs are runtime-managed via the
            // tensor pool's ARC), so leaving the bumped clone bare
            // would leak one ref per Linear call (7+ per block, 16
            // blocks → 100+ leaked refs per generated token).
            if (useHaxeMatmul()) {
                y = Q4Matmul.matmul(qweight, x, pool);
            } else {
                var xClone = x.clone();
                y = qweight.matmulXTQThreaded(xClone, 0);
                xClone.free();
            }
        } else {
            y = x.matmulT(weight);
        }
        if (bias != null) {
            // `y` is a fresh contiguous owning tensor from matmulT or
            // matmulXTQThreaded. Add bias in place — saves one
            // [seq, out_features] F32 alloc per Linear call. Cold for
            // Llama (no bias on any linear) but warm for Qwen2 (bias on
            // Q/K/V) and most encoder models.
            y.addInto(bias);
            return y;
        }
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
