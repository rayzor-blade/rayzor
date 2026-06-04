package nue;

import rayzor.ds.Tensor;
import rayzor.ds.QTensor;

/**
 * Token embedding lookup.
 *
 * Owns a `[vocab_size, hidden_size]` weight table and produces
 * `[seq_len, hidden_size]` activations by indexing rows for each
 * token ID. Implemented via `Tensor.gatherRows` — a constant-time
 * row copy per token, no matmul.
 *
 * Phase 4b: when constructed via `Embedding.fromQuant`, the layer
 * holds a `QTensor` (Q6_K) weight and runs `gatherRowsQ6K` so the
 * `[vocab, hidden]` table never materialises in F32 — the only rows
 * dequantised are the ones named by the current step's token IDs.
 * Memory drops ~5× vs the dequant path (Q6_K is ~6.5 bpw) and
 * output is byte-identical to "dequant the whole table then gather"
 * because both paths route through the same Q6_K block decoder.
 * The F32 `weight` slot holds a 1×1 sentinel (kept non-null to keep
 * the class layout stable — see `nue.Linear`).
 *
 * The same module is reused for the language-model head when
 * `tie_word_embeddings` is true: feed the final hidden state through
 * a linear with `weight.T` to produce vocabulary logits. We expose
 * `embedTable()` so the LM head can borrow the storage without
 * copying it. In the Q6_K case, `embedTable()` returns the `QTensor`
 * (callers must check `qweight != null` and route to
 * `Linear.fromQuant`); in the F32 case it returns the `Tensor`.
 */
class Embedding implements Module {
    public var weight:Tensor;
    public var qweight:Null<QTensor>;
    public var vocabSize:Int;
    public var hiddenSize:Int;
    public var paramName:String;

    public function new(weight:Tensor, vocabSize:Int, hiddenSize:Int, paramName:String) {
        this.weight = weight;
        this.qweight = null;
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.paramName = paramName;
    }

    /**
     * Build an Embedding whose weight is a `QTensor` (Q6_K storage).
     * Lookup runs the Q6_K-aware row gather; no F32 `[vocab, hidden]`
     * table is allocated. The F32 `weight` slot is filled with a 1×1
     * sentinel to keep the class layout stable (see class doc).
     */
    public static function fromQuant(qweight:QTensor, vocab:Int, hidden:Int, paramName:String):Embedding {
        // Bare `F32` resolves via TAST enum-variant disambiguation; see
        // nue.Linear.fromQuant for the same pattern.
        var sentinel = Tensor.zeros([1, 1], F32);
        var e = new Embedding(sentinel, vocab, hidden, paramName);
        e.qweight = qweight;
        return e;
    }

    /**
     * Look up `tokenIds` (length seq_len) in the embedding table.
     * Returns `[seq_len, hidden_size]`.
     *
     * In the Q6_K case this dequantises only the named rows. The
     * output is byte-identical to a full `qweight.dequant()` followed
     * by an F32 `gatherRows`, because both paths route through the
     * same Q6_K block decoder.
     */
    public function lookup(tokenIds:Array<Int>):Tensor {
        if (qweight != null) {
            return qweight.gatherRowsQ6K(tokenIds);
        }
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
        for (i in 0...n) ids.push(Std.int(x.getFlat(i)));
        return lookup(ids);
    }

    public function parameters():Array<NamedTensor> {
        if (qweight != null) {
            // Quantised — `weight` is a 1×1 sentinel, not a real parameter.
            return [];
        }
        return [{ name: paramName, tensor: weight }];
    }

    /**
     * Borrow the underlying weight table for LM-head weight tying.
     * Returns the `QTensor` in the Q6_K case and the `Tensor` in the
     * F32 case — callers should inspect `qweight` first and route to
     * `Linear.fromQuant` when non-null, otherwise wrap as F32 `Linear`.
     * Typed `Dynamic` so the same accessor covers both shapes without
     * forcing a precarious cast at every call site.
     */
    public function embedTable():Dynamic {
        if (qweight != null) return qweight;
        return weight;
    }
}
