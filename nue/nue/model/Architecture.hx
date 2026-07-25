package nue.model;

/**
 * Known model architecture families.
 *
 * A CLOSED set for typed framework logic (chat-template selection,
 * arch-specific defaults). It is deliberately NOT the source of truth for
 * architecture dispatch: `ModelMetadata.architecture` stays a `String` so
 * `ArchRegistry` remains an OPEN set that users extend with custom
 * architectures. `Unknown` covers anything outside this set.
 */
enum Architecture {
    Llama;
    Qwen2;
    Mistral;
    Gemma;
    GPT2;
    Falcon;
    Phi;
    Bert;
    Unknown;
}

/**
 * Helpers for `Architecture` (plain Haxe enums can't carry methods).
 */
class Architectures {
    /**
     * Map a `ModelMetadata.architecture` string to a known family, or
     * `Unknown`. Case-insensitive; accepts the common aliases a GGUF or
     * safetensors config carries.
     */
    public static function fromString(s:String):Architecture {
        if (s == null) return Unknown;
        var a = s.toLowerCase();
        if (a == "llama") return Llama;
        if (a == "qwen2" || a == "qwen") return Qwen2;
        if (a == "mistral") return Mistral;
        if (a == "gemma" || a == "gemma2") return Gemma;
        if (a == "gpt2") return GPT2;
        if (a == "falcon") return Falcon;
        if (a == "phi" || a == "phi2" || a == "phi3") return Phi;
        if (a == "bert") return Bert;
        return Unknown;
    }

    /** Canonical lowercase family name (round-trips `fromString`). */
    public static function toName(a:Architecture):String {
        var n = "unknown";
        switch (a) {
            case Llama: n = "llama";
            case Qwen2: n = "qwen2";
            case Mistral: n = "mistral";
            case Gemma: n = "gemma";
            case GPT2: n = "gpt2";
            case Falcon: n = "falcon";
            case Phi: n = "phi";
            case Bert: n = "bert";
            case Unknown: n = "unknown";
        }
        return n;
    }
}
