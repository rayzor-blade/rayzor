package nue.loader;

import nue.Module;
import nue.model.ModelLoader;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import rayzor.ds.Tensor;

/**
 * ONNX (`*.onnx`) loader — protobuf-encoded graph + initialisers.
 *
 * Substantially more involved than GGUF/safetensors because the
 * format encodes an arbitrary computation graph, not just a weight
 * bundle. For LLM inference we only consume the initialisers (model
 * weights) and rely on the architecture builder to assemble the
 * computation — we do NOT plan to execute arbitrary ONNX graphs.
 *
 * **Strategy when this is implemented:**
 *   1. Read protobuf header to extract `ModelProto.graph.initializer`
 *      entries (the named weight tensors).
 *   2. Translate ONNX dtype tags (1=F32, 10=F16, etc.) and shape
 *      info into our `Tensor` representation.
 *   3. Map HF/PyTorch-style names to the GGUF canonical scheme.
 *
 * **Status:** stub. Lowest priority of the three loaders because
 * GGUF + safetensors cover the entire HF model hub. ONNX support
 * would mainly serve users who already standardised on the format
 * (often for edge deployment).
 *
 * **Minimal protobuf decoder.** Rather than ship a full protobuf
 * library, we'd hand-roll a small varint + wire-tag reader scoped
 * to the subset of `ModelProto` we actually need. ~500 LOC.
 */
class ONNXLoader implements ModelLoader {
    public function new() {}

    public function readMetadata(_:String):ModelMetadata {
        throw "ONNXLoader: not yet implemented. Use GGUF for LLM inference.";
    }

    public function readNamedTensors(_:String):NamedTensorMap {
        throw "ONNXLoader: not yet implemented.";
    }

    public function load(_:String):Module {
        throw "ONNXLoader: not yet implemented.";
    }
}
