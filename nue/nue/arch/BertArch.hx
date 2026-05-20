package nue.arch;

import nue.Module;
import nue.Linear;
import nue.Embedding;
import nue.EncoderModel;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import nue.transformer.LayerNorm;
import nue.transformer.MultiHeadAttention;
import nue.transformer.GeluFFN;
import nue.transformer.TransformerBlock;
import rayzor.ds.Tensor;

/**
 * Bidirectional encoder builder — BERT / RoBERTa / DeBERTa family.
 *
 * The encoder composes:
 *   - token embedding + segment embedding (BERT) + absolute position
 *     embedding, summed before block 0
 *   - N × TransformerBlock(LayerNorm, MultiHeadAttention, LayerNorm, GeluFFN)
 *   - final LayerNorm
 *   - optional task heads (MLM, classification) — those live in
 *     downstream user code or future `nue.head.*` modules
 *
 * BERT weight naming differs from GGUF; the loader is responsible
 * for normalising into a canonical scheme this builder reads against.
 *
 * **NOTE:** this builder is currently a stub demonstrating that the
 * `ArchRegistry` + `EncoderModel` design supports encoders. Wire-up
 * of the BERT-specific embedding sum (token + segment + position) and
 * the per-format weight name normalisation lands when there's a
 * concrete request — pure-Haxe + reusing the existing nue primitives,
 * no kernel additions needed.
 */
class BertArch implements ArchBuilder {
    public function new() {}

    public function name():String {
        return "bert";
    }

    public function validate(meta:ModelMetadata):Bool {
        if (meta.numHeads <= 0 || meta.numKvHeads != meta.numHeads) return false;
        if (meta.headDim * meta.numHeads != meta.hiddenSize) return false;
        if (meta.numLayers <= 0 || meta.vocabSize <= 0) return false;
        return true;
    }

    public function build(meta:ModelMetadata, weights:NamedTensorMap):Module {
        throw "nue.arch.BertArch.build: not yet implemented — see class docs";
    }
}
