package nue;

import rayzor.ds.Tensor;

/**
 * Token embedding lookup.
 *
 * Owns a `[vocab_size, hidden_size]` weight table and produces
 * `[seq_len, hidden_size]` activations by indexing rows for each
 * token ID. Implemented via `Tensor.gatherRows` — a constant-time
 * row copy per token, no matmul.
 *
 * The same module is reused for the language-model head when
 * `tie_word_embeddings` is true: feed the final hidden state through
 * a linear with `weight.T` to produce vocabulary logits. We expose
 * `embedTable()` so the LM head can borrow the storage without
 * copying it.
 */
class Embedding implements Module {
    public var weight:Tensor;
    public var vocabSize:Int;
    public var hiddenSize:Int;
    public var paramName:String;

    public function new(weight:Tensor, vocabSize:Int, hiddenSize:Int, paramName:String) {
        this.weight = weight;
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.paramName = paramName;
    }

    /**
     * Look up `tokenIds` (length seq_len) in the embedding table.
     * Returns `[seq_len, hidden_size]`.
     */
    public function lookup(tokenIds:Array<Int>):Tensor {
        return weight.gatherRows(tokenIds);
    }

    /**
     * Module interface compliance — accepts a `Tensor` to play nicely
     * with the generic `Module.forward(Tensor)` signature. The input
     * is expected to be a 1-D integer tensor with vocabulary indices
     * stored as I32 — converted to an Array<Int> here.
     *
     * Production code should call `lookup(tokenIds)` directly to skip
     * the marshalling, but `forward` is provided so an `Embedding`
     * can drop into a generic Sequential / pipeline.
     */
    public function forward(x:Tensor):Tensor {
        var n = x.numel();
        var ids = [];
        for (i in 0...n) ids.push(Std.int(x.get([i])));
        return lookup(ids);
    }

    public function parameters():Array<NamedTensor> {
        return [{ name: paramName, tensor: weight }];
    }

    /** Borrow the underlying weight table for LM-head weight tying. */
    public function embedTable():Tensor {
        return weight;
    }
}
