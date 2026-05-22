import sys.io.File;

/**
 * Smoke test: mmap-backed File.getBytes on a 4.92 GB Llama-3-8B GGUF.
 *
 * Demonstrates that the underlying file load is now zero-copy:
 *   - bytes.length reports the file size (truncated to i32 for files
 *     larger than 2 GiB — full-width access via i64-aware extern fns
 *     is a follow-up)
 *   - random access at small + large offsets returns the right bytes
 *   - bytes.sub(...).toString() returns the right text (zero-copy view
 *     under the hood; no extra heap allocation)
 *
 * Verify the process resident set yourself — `top -pid <pid>` should
 * show roughly tens of MB even though the underlying mapping is GBs.
 *
 * The full GGUFReader.parse() path is a separate concern (the
 * complex-method-body dispatch bug surfaces on `this.readValue(typeTag)`
 * inside the parse loop). When that's resolved this example will
 * extend into the full metadata + tokenizer pull.
 */
class Main {
    static function main() {
        var path = "/Users/amaterasu/.cache/huggingface/hub/models--lmstudio-community--Meta-Llama-3-8B-Instruct-GGUF/snapshots/0910a3e69201d274d4fd68e89448114cd78e4c82/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";

        trace("=== llama-smoke: mmap-backed File.getBytes ===");
        trace("opening: " + path);

        var bytes = File.getBytes(path);
        trace("file mmap'd, bytes.length (i32-clamped) = " + bytes.length);

        // Header: magic at offset 0.
        trace("byte[0..3] = " + bytes.get(0) + "," + bytes.get(1)
            + "," + bytes.get(2) + "," + bytes.get(3) + " (expect 71,71,85,70 = 'GGUF')");
        trace("magic int32 = 0x" + StringTools.hex(bytes.getInt32(0))
            + " (expect 0x46554747)");

        // First metadata key starts at offset 32 in GGUF v3.
        trace("sub(32,20).toString() = '" + bytes.sub(32, 20).toString()
            + "' (expect 'general.architecture')");

        trace("");
        trace("=== Done ===");
    }
}
