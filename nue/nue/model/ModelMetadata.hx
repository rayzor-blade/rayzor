package nue.model;

/**
 * Architecture-agnostic metadata extracted from a model file.
 *
 * Every supported file format (GGUF, ONNX, safetensors, PyTorch) parses
 * down to a `ModelMetadata` plus a per-tensor data dictionary. The
 * `architecture` string then selects which `nue.arch.*` builder
 * instantiates the matching Module tree, and the rest of the fields
 * provide the shape parameters that builder needs.
 *
 * Not every architecture uses every field — `numKvHeads` collapses to
 * `numHeads` for non-GQA models, `intermediateSize` doesn't apply to
 * MoE blocks, etc. The arch builder is responsible for translating
 * this generic config into its specific class instances. Extra
 * format-specific knobs (e.g. ONNX op-set version, GGUF quant scheme)
 * are passed alongside via `extras` rather than expanding this struct.
 */
typedef ModelMetadata = {
    /**
     * Architecture family name as declared by the file: `"llama"`,
     * `"gpt2"`, `"mistral"`, `"qwen2"`, `"falcon"`, `"phi"`, …
     * Lowercased; the `nue.arch` registry is keyed on this string.
     */
    var architecture:String;

    var hiddenSize:Int;
    var intermediateSize:Int;
    var numLayers:Int;
    var numHeads:Int;
    var numKvHeads:Int;
    var headDim:Int;
    var vocabSize:Int;
    var maxSeqLen:Int;
    var normEps:Float;
    var ropeBase:Float;
    var tieWordEmbeddings:Bool;

    /**
     * Per-architecture extras keyed by name — anything that doesn't
     * fit the generic schema. Use sparingly; if a knob is shared by
     * many architectures, promote it to a real field.
     */
    @:optional var extras:Map<String, String>;
}
