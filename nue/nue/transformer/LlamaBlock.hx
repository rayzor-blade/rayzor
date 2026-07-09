package nue.transformer;

import nue.Module.NamedTensor;
import rayzor.ds.Tensor;

/**
 * Concrete Llama/Mistral/Qwen pre-norm block.
 *
 * The generic `TransformerBlock` stores children as `Module`, which is useful
 * for architecture reuse but forces interface dispatch in the llama decode hot
 * path. This class keeps the same tensor ownership pattern with concrete child
 * types so Rayzor can lower direct calls through RMSNorm, GQAttention, and
 * SwiGLU before reaching the explicit SIMD kernels.
 */
class LlamaBlock {
    public var attnNorm:RMSNorm;
    public var attn:GQAttention;
    public var ffnNorm:RMSNorm;
    public var ffn:SwiGLU;
    public var blockName:String;
    static var _debugShapes:Int = 0;

    public function new(
        attnNorm:RMSNorm,
        attn:GQAttention,
        ffnNorm:RMSNorm,
        ffn:SwiGLU,
        name:String
    ) {
        this.attnNorm = attnNorm;
        this.attn = attn;
        this.ffnNorm = ffnNorm;
        this.ffn = ffn;
        this.blockName = name;
    }

    static function debugShapeMode():Int {
        if (_debugShapes == 0) {
            var v = Sys.getEnv("RAYZOR_DUMP_BLOCK_SHAPES");
            if (v == null || v == "0" || v == "" || v == "false") {
                _debugShapes = 2;
            } else if (v == "trace") {
                _debugShapes = 3;
            } else {
                _debugShapes = 1;
            }
        }
        return _debugShapes;
    }

    static function shapeStr(t:Tensor):String {
        if (t == null) return "null";
        var s = t.shape();
        var out = "[";
        for (i in 0...s.length) {
            if (i > 0) out += ",";
            out += "" + s[i];
        }
        return out + "]";
    }

    static function checkShape(blockName:String, stage:String, t:Tensor, hiddenSize:Int, traceAll:Bool):Void {
        var s = t.shape();
        var ok = s.length == 2 && s[0] > 0 && s[0] <= 4096 && s[1] == hiddenSize;
        if (traceAll || !ok) {
            Sys.println("[block-shape] " + blockName + " " + stage + " " + shapeStr(t));
        }
    }

    public function forward(x:Tensor):Tensor {
        var dbg = debugShapeMode();
        var traceShapes = dbg == 3;
        var checkShapes = dbg == 1 || dbg == 3;
        var hiddenSize = attn.numQHeads * attn.headDim;
        var label:String = this.blockName;
        if (checkShapes) checkShape(label, "enter x=", x, hiddenSize, traceShapes);
        var xClone1 = x.clone();
        var t = attnNorm.forward(xClone1);
        xClone1.free();
        if (checkShapes) {
            checkShape(label, "after_attn_norm x=", x, hiddenSize, traceShapes);
            checkShape(label, "after_attn_norm norm=", t, hiddenSize, traceShapes);
        }
        var attnOut = attn.forward(t);
        t.free();
        if (checkShapes) {
            checkShape(label, "before_attn_residual x=", x, hiddenSize, traceShapes);
            checkShape(label, "before_attn_residual attn=", attnOut, hiddenSize, traceShapes);
        }
        x.addInto(attnOut);
        attnOut.free();
        var h1 = x;

        var xClone2 = h1.clone();
        var t2 = ffnNorm.forward(xClone2);
        xClone2.free();
        if (checkShapes) {
            checkShape(label, "after_ffn_norm h=", h1, hiddenSize, traceShapes);
            checkShape(label, "after_ffn_norm norm=", t2, hiddenSize, traceShapes);
        }
        var ffnOut = ffn.forward(t2);
        t2.free();
        if (checkShapes) {
            checkShape(label, "before_ffn_residual h=", h1, hiddenSize, traceShapes);
            checkShape(label, "before_ffn_residual ffn=", ffnOut, hiddenSize, traceShapes);
        }
        h1.addInto(ffnOut);
        ffnOut.free();
        if (checkShapes) checkShape(label, "exit h=", h1, hiddenSize, traceShapes);
        return h1;
    }

    public function parameters():Array<NamedTensor> {
        // LlamaBlock is used only after the GGUF loader has already wired
        // concrete child modules. The recursive parameter walk is cold
        // introspection, and the current compiler sometimes resolves the
        // `NamedTensor.name` typedef field ambiguously inside this concrete
        // hot-path class. Keep inference bundles trap-free while that resolver
        // bug is handled separately.
        return [];
    }
}
