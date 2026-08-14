package nue.arch;

/**
 * Full built-in registry helper.
 *
 * `ArchRegistry.withDefaults()` intentionally stays Llama-family only so
 * latency-sensitive LLM examples do not cold-lower BERT encoder code. Call
 * this helper when an application wants every built-in architecture.
 */
class BuiltinArchRegistry {
    public static function withDefaults():ArchRegistry {
        var r = ArchRegistry.withDefaults();
        r.register("bert", new BertArch());
        return r;
    }
}
