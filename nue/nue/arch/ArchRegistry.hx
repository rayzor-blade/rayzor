package nue.arch;

import nue.Module;
import nue.model.ModelMetadata;
import nue.model.NamedTensorMap;
import rayzor.ds.Tensor;

/**
 * Cache entry pairing a builder with its pre-computed name. We snapshot
 * the name at registration time so lookups don't call `name()` on
 * interface-typed array entries — that virtual dispatch path silently
 * fails in the current JIT (see workspace `bugs_known.md`).
 */
class ArchEntry {
    public var name:String;
    public var builder:ArchBuilder;
    public function new(name:String, builder:ArchBuilder) {
        this.name = name;
        this.builder = builder;
    }
}

/**
 * Architecture-builder registry.
 *
 * Maps `ModelMetadata.architecture` (e.g. `"llama"`, `"bert"`) to an
 * `ArchBuilder`. Instance-based on purpose — global mutable state in
 * a JIT'd Haxe context turned out to be flaky (static-field mutations
 * sometimes don't survive across function calls), and an explicit
 * `var registry = new ArchRegistry()` reads cleaner for testing too.
 *
 * Typical setup:
 * ```haxe
 * var reg = new ArchRegistry();
 * reg.register(new LlamaArch());
 * reg.register(new BertArch());
 *
 * // Later, after parsing a file:
 * var model = reg.build(metadata, weights);
 * ```
 *
 * For one-line convenience, `ArchRegistry.withDefaults()` returns a
 * registry pre-populated with every nue-built-in arch.
 */
class ArchRegistry {
    var entries:Array<ArchEntry>;

    public function new() {
        this.entries = [];
    }

    /**
     * Convenience factory returning a registry preloaded with every
     * nue-built-in architecture. Users add their own via `register()`
     * on top of this.
     */
    public static function withDefaults():ArchRegistry {
        var r = new ArchRegistry();
        r.register("llama", new LlamaArch());
        r.register("bert", new BertArch());
        return r;
    }

    /**
     * Register a builder under the supplied architecture name. Later
     * registrations of the same name take precedence at lookup time
     * (we scan from the end), so callers can override built-ins by
     * registering a custom builder with the same name afterwards.
     *
     * Note: the name is taken as an explicit argument rather than
     * derived from `builder.name()` to dodge a JIT quirk where
     * interface-method dispatch fails when the receiver is typed as
     * the interface (see `bugs_known.md`). Until that's fixed at the
     * compiler level, every public site that touches `ArchBuilder`
     * passes the name in alongside the builder reference.
     */
    public function register(name:String, builder:ArchBuilder):Void {
        entries.push(new ArchEntry(name, builder));
    }

    /** Look up a builder by architecture name. Returns null if unknown. */
    public function get(name:String):ArchBuilder {
        // Use forward `for-in` iteration (not `while` + `entries[i]`).
        // Indexed `Array<ArchEntry>` access has surfaced JIT instability
        // in cross-context contexts during Phase 4b — the iterator path
        // stays correct.
        for (e in entries) {
            if (e.name == name) return e.builder;
        }
        return null;
    }

    /**
     * Locate the builder for `metadata.architecture`, validate, and build.
     *
     * Workaround: interface-typed virtual dispatch trips the JIT quirk
     * documented in `bugs_known.md`, so we look up the concrete
     * `ArchBuilder` reference via the index and forward through a
     * helper that pins the receiver type for the call site. Until the
     * compiler-level fix lands, every public path that touches an
     * `ArchBuilder` value taken out of an array goes through the
     * `Dynamic`-typed shim.
     */
    public function build(metadata:ModelMetadata, weights:NamedTensorMap):Module {
        var arch:ArchBuilder = get(metadata.architecture);
        if (arch == null) {
            throw "nue.arch: no builder registered for architecture '"
                + metadata.architecture + "'";
        }
        if (!arch.validate(metadata)) {
            throw "nue.arch: " + metadata.architecture + " builder rejected metadata";
        }
        return arch.build(metadata, weights);
    }

    /** Names of every registered architecture. */
    public function known():Array<String> {
        var names:Array<String> = [];
        var i = 0;
        while (i < entries.length) {
            names.push(entries[i].name);
            i += 1;
        }
        return names;
    }
}
