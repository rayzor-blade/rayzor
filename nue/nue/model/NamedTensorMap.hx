package nue.model;

import haxe.ds.StringMap;
import rayzor.ds.Tensor;
import rayzor.ds.QTensor;

/**
 * Name → Tensor lookup used by loaders to hand model weights to arch
 * builders. Backed by `StringMap<Tensor>` — O(1) average lookup.
 *
 * `names` is preserved as an ordered list so iteration matches
 * insertion order (mirrors GGUF's tensor index order, which is what
 * model builders expect when walking layer-by-layer).
 *
 * Phase 4b: an optional parallel `qtensorByName` side channel lets
 * loaders ship quantised weights through without dequantising. When a
 * name has a QTensor entry, the regular `tensorByName` slot is a small
 * placeholder (or null) and the arch builder is expected to construct
 * a quantisation-aware module (e.g. `Linear.fromQuant`) instead of the
 * F32 path.
 *
 * Names follow the GGUF canonical convention (`token_embd.weight`,
 * `blk.{L}.attn_q.weight`, …) regardless of source file format.
 */
class NamedTensorMap {
    public var names:Array<String>;
    public var tensorByName:StringMap<Tensor>;
    public var qtensorByName:StringMap<QTensor>;

    public function new() {
        this.names = [];
        this.tensorByName = new StringMap<Tensor>();
        this.qtensorByName = new StringMap<QTensor>();
    }

    public inline function size():Int {
        return names.length;
    }

    public function set(name:String, tensor:Tensor):Void {
        if (!tensorByName.exists(name) && !qtensorByName.exists(name)) {
            names.push(name);
        }
        tensorByName.set(name, tensor);
    }

    /** Register a QTensor-backed weight under `name`. Mutually exclusive
        with `set`: a name has either a Tensor or a QTensor, not both. */
    public function setQuant(name:String, qt:QTensor):Void {
        if (!tensorByName.exists(name) && !qtensorByName.exists(name)) {
            names.push(name);
        }
        qtensorByName.set(name, qt);
    }

    /** O(1) lookup. Returns null if the name is not present. */
    public function get(name:String):Tensor {
        return tensorByName.get(name);
    }

    public function getQuant(name:String):QTensor {
        return qtensorByName.get(name);
    }

    public function exists(name:String):Bool {
        return tensorByName.exists(name) || qtensorByName.exists(name);
    }

    public function isQuantised(name:String):Bool {
        return qtensorByName.exists(name);
    }
}
