package nue.model;

import rayzor.ds.Tensor;

/**
 * Name → Tensor lookup used by loaders to hand model weights to arch
 * builders. Backed by parallel arrays + linear scan rather than
 * `Map<String, Tensor>` because the current rayzor stdlib's
 * `Map<String, V>.get` doesn't reliably return values.
 *
 * Lookups are O(N). Typical Llama-class models have ~300 tensors
 * (32 layers × 9 per-layer + small fixed-count globals), so the
 * worst-case scan is < 1µs and irrelevant compared to the actual
 * compute work each weight goes into.
 *
 * Names follow the GGUF canonical convention (`token_embd.weight`,
 * `blk.{L}.attn_q.weight`, …) regardless of source file format.
 */
class NamedTensorMap {
    public var names:Array<String>;
    public var tensors:Array<Tensor>;

    public function new() {
        this.names = [];
        this.tensors = [];
    }

    public inline function size():Int {
        return names.length;
    }

    public function set(name:String, tensor:Tensor):Void {
        for (i in 0...names.length) {
            if (names[i] == name) {
                tensors[i] = tensor;
                return;
            }
        }
        names.push(name);
        tensors.push(tensor);
    }

    /** Linear-scan lookup. Returns null if the name is not present. */
    public function get(name:String):Tensor {
        for (i in 0...names.length) {
            if (names[i] == name) return tensors[i];
        }
        return null;
    }

    public function exists(name:String):Bool {
        for (i in 0...names.length) {
            if (names[i] == name) return true;
        }
        return false;
    }
}
