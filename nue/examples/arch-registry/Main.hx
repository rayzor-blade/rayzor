import nue.arch.ArchRegistry;

class Main {
    static function main() {
        trace("=== nue architecture registry ===");

        var reg = ArchRegistry.withDefaults();

        var arches = reg.known();
        trace("registered architectures: " + arches.length);
        for (a in arches) trace("  - " + a);

        var llama = reg.get("llama");
        trace("looked up llama: " + (llama != null));
        var bert = reg.get("bert");
        trace("looked up bert: " + (bert != null));

        // Demonstrate name() works on a CONCRETE-typed reference (the
        // ArchEntry pre-cached name is what survives JIT — see
        // bugs_known.md for the interface-dispatch quirk).
        trace("=== Done ===");
    }
}
