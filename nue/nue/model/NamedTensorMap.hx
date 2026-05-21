package nue.model;

import haxe.ds.StringMap;
import rayzor.ds.Tensor;

/**
 * Name → Tensor lookup used by loaders to hand model weights to arch
 * builders. Backed by `StringMap<Tensor>` — O(1) average lookup.
 *
 * `names` is preserved as an ordered list so iteration matches
 * insertion order (mirrors GGUF's tensor index order, which is what
 * model builders expect when walking layer-by-layer).
 *
 * Names follow the GGUF canonical convention (`token_embd.weight`,
 * `blk.{L}.attn_q.weight`, …) regardless of source file format.
 */
class NamedTensorMap {
    public var names:Array<String>;
    public var tensorByName:StringMap<Tensor>;

    public function new() {
        this.names = [];
        this.tensorByName = new StringMap<Tensor>();
    }

    public inline function size():Int {
        return names.length;
    }

    public function set(name:String, tensor:Tensor):Void {
        if (!tensorByName.exists(name)) {
            names.push(name);
        }
        tensorByName.set(name, tensor);
    }

    /** O(1) lookup. Returns null if the name is not present. */
    public function get(name:String):Tensor {
        return tensorByName.get(name);
    }

    public function exists(name:String):Bool {
        return tensorByName.exists(name);
    }
}
