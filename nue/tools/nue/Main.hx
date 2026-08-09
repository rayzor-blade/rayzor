import sys.FileSystem;
import sys.io.File;
import nue.loader.GGUFReader;

/**
 * `nue` — the model-side CLI.
 *
 * Exists because the prefill-graph pipeline was an undocumented two-step
 * (a python/coremltools script + `xcrun coremlc compile`) that nothing in the
 * repo invoked, so a model either happened to have artifacts beside it or
 * silently fell back to CPU prefill with no diagnostic. `prefill status` makes
 * that visible and `prefill author` makes it reproducible.
 *
 *   nue info    <model.gguf>
 *   nue prefill status <model.gguf> [ctx]
 *   nue prefill author <model.gguf> [bucket ...]
 */
class Main {
    // GGUF dtype id -> name. Only the ids nue can encounter are named; the
    // rest print as the raw id rather than a wrong guess.
    static function dtypeName(t:Int):String {
        switch (t) {
            case 0: return "F32";
            case 1: return "F16";
            case 2: return "Q4_0";
            case 3: return "Q4_1";
            case 6: return "Q5_0";
            case 7: return "Q5_1";
            case 8: return "Q8_0";
            case 10: return "Q2_K";
            case 11: return "Q3_K";
            case 12: return "Q4_K";
            case 13: return "Q5_K";
            case 14: return "Q6_K";
            case 15: return "Q8_K";
            default: return "dtype" + t;
        }
    }

    /** Whole-file read, as `GGUFLoader` does.

        A prefix read would be far cheaper — this pulls 4.4 GB to print a dozen
        fields — but `GGUFReader` over a TRUNCATED buffer reads past the end and
        SIGSEGVs rather than throwing, so `try`/grow-the-prefix cannot recover:
        the process is already gone. Making `info` cheap needs a reader that
        bounds-checks against the buffer, which is a change to GGUFReader, not
        to this tool. Correctness first. */
    static function readHeader(path:String):GGUFReader {
        return new GGUFReader(File.getBytes(path));
    }

    static function metaStringOr(r:GGUFReader, key:String, dflt:String):String {
        if (r.findMeta(key) == null) return dflt;
        return r.metaString(key);
    }

    static function metaIntOr(r:GGUFReader, key:String, dflt:Int):Int {
        if (r.findMeta(key) == null) return dflt;
        return r.metaInt(key);
    }

    static function info(path:String):Int {
        var r = readHeader(path);
        var arch = metaStringOr(r, "general.architecture", "?");
        Sys.println("file        " + path);
        Sys.println("gguf        v" + r.version + "  tensors=" + r.nTensors + "  meta_kv=" + r.nMetaKv);
        Sys.println("arch        " + arch);
        Sys.println("name        " + metaStringOr(r, "general.name", "?"));

        var hidden = metaIntOr(r, arch + ".embedding_length", 0);
        var layers = metaIntOr(r, arch + ".block_count", 0);
        var heads = metaIntOr(r, arch + ".attention.head_count", 0);
        var kvHeads = metaIntOr(r, arch + ".attention.head_count_kv", 0);
        var ctx = metaIntOr(r, arch + ".context_length", 0);
        var headDim = (heads > 0) ? Std.int(hidden / heads) : 0;
        Sys.println("shape       hidden=" + hidden + " layers=" + layers
            + " heads=" + heads + " kv_heads=" + kvHeads + " head_dim=" + headDim);
        Sys.println("context     " + ctx);

        // Identity is NOT the arch string: Mistral, Llama-2 and Llama-3 all
        // declare "llama" and need different prompt formats. Report the
        // tokenizer evidence the template selection actually keys on.
        var tokModel = metaStringOr(r, "tokenizer.ggml.model", "?");
        var hasMerges = (r.findMeta("tokenizer.ggml.merges") != null);
        var hasScores = (r.findMeta("tokenizer.ggml.scores") != null);
        Sys.println("tokenizer   model=" + tokModel
            + " merges=" + (hasMerges ? "yes" : "NO")
            + " scores=" + (hasScores ? "yes" : "no"));
        if (!hasMerges && hasScores) {
            Sys.println("            ^ sentencepiece (unigram). nue's BPE path needs merges;");
            Sys.println("              without them encode degrades toward character-level.");
        }

        // Quantisation census straight from the tensor table — what the model
        // IS, independent of which kernels a run happens to take.
        // Parallel arrays, not a Map: StringMap has open defects here
        // (get(missing) yields a raw 0, keys() panics) and the set is tiny.
        var ids:Array<Int> = [];
        var num:Array<Int> = [];
        for (i in 0...r.tensorInfos.length) {
            var t = r.tensorInfos[i].dtype;
            var at = -1;
            for (j in 0...ids.length) if (ids[j] == t) at = j;
            if (at < 0) {
                ids.push(t);
                num.push(1);
            } else {
                num[at] = num[at] + 1;
            }
        }
        var parts:Array<String> = [];
        for (i in 0...ids.length) parts.push(dtypeName(ids[i]) + "=" + num[i]);
        Sys.println("quant       " + parts.join("  "));
        return 0;
    }

    /** Buckets nue would look for. Artifacts are FIXED-shape (one per bucket),
        so a prompt is served by the smallest bucket >= its length. */
    static function defaultBuckets(ctx:Int):Array<Int> {
        var all = [128, 512, 1024, 2048];
        var out:Array<Int> = [];
        for (i in 0...all.length) if (ctx <= 0 || all[i] <= ctx) out.push(all[i]);
        if (out.length == 0) out.push(128);
        return out;
    }

    static function stemOf(path:String):String {
        var slash = path.lastIndexOf("/");
        var file = slash >= 0 ? path.substr(slash + 1) : path;
        var dot = file.lastIndexOf(".gguf");
        return dot > 0 ? file.substr(0, dot) : file;
    }

    static function dirOf(path:String):String {
        var slash = path.lastIndexOf("/");
        return slash >= 0 ? path.substr(0, slash) : ".";
    }

    static function prefillStatus(path:String, ctx:Int):Int {
        var r = readHeader(path);
        var arch = metaStringOr(r, "general.architecture", "?");
        var hidden = metaIntOr(r, arch + ".embedding_length", 0);
        var layers = metaIntOr(r, arch + ".block_count", 0);
        var dir = dirOf(path);
        var stem = stemOf(path);
        Sys.println("model       " + stem);
        Sys.println("arch        " + arch
            + (arch == "llama" || arch == "qwen2" ? "  (prefill graph supported)"
                                                 : "  (prefill graph NOT wired for this arch)"));

        // fp16 weights are baked per bucket, so each artifact is roughly the
        // parameter count in halves. This is why a 7B is impractical today:
        // ~2.5 GB at 1B scales linearly.
        var ffn = metaIntOr(r, arch + ".feed_forward_length", 0);
        var perLayer = 4.0 * hidden * hidden + 3.0 * hidden * ffn;
        var approxBytes = perLayer * layers * 2.0;
        var found = 0;
        var buckets = defaultBuckets(ctx);
        for (i in 0...buckets.length) {
            var p = dir + "/" + stem + ".prefill_s" + buckets[i] + ".mlmodelc";
            var have = FileSystem.exists(p);
            if (have) found++;
            Sys.println("bucket s" + buckets[i] + (have ? "   PRESENT  " : "   missing  ") + p);
        }
        Sys.println("est/bucket  " + Std.int(approxBytes / 1048576) + " MB (fp16, baked)");
        if (found == 0) {
            Sys.println("");
            Sys.println("No prefill artifacts: this model falls back to CPU prefill.");
            Sys.println("Author them with:  nue prefill author " + path);
        }
        return found > 0 ? 0 : 1;
    }

    static function prefillAuthor(path:String, buckets:Array<Int>):Int {
        var repo = Sys.getEnv("RAYZOR_REPO");
        if (repo == null || repo == "") repo = "../../..";
        var script = repo + "/nue-plugins/examples/llama_prefill_author.py";
        var venv = repo + "/nue-plugins/examples/mlvenv/bin/python";
        var dir = dirOf(path);
        var stem = stemOf(path);

        if (!FileSystem.exists(script)) {
            Sys.println("nue: author script not found at " + script);
            Sys.println("     set RAYZOR_REPO to the repo root.");
            return 2;
        }
        // The toolchain is python/coremltools — deliberately NOT reimplemented
        // in Haxe: authoring emits a CoreML MIL program, which is coremltools'
        // serialization format. The CLI owns discovery, naming, bucket policy
        // and verification; the converter stays where it is.
        if (!FileSystem.exists(venv)) {
            Sys.println("nue: CoreML toolchain missing (" + venv + ")");
            Sys.println("     python3 -m venv " + repo + "/nue-plugins/examples/mlvenv");
            Sys.println("     " + venv + " -m pip install -r " + repo + "/nue-plugins/examples/requirements.txt");
            return 2;
        }
        var cmd = q(venv) + " " + q(script) + " " + q(path) + " " + q(dir);
        for (i in 0...buckets.length) cmd += " " + buckets[i];
        Sys.println("nue: authoring " + buckets.length + " bucket(s) for " + stem);
        var rc = Sys.command(cmd);
        if (rc != 0) {
            Sys.println("nue: author failed (rc=" + rc + ")");
            return rc;
        }
        for (i in 0...buckets.length) {
            var pkg = dir + "/" + stem + ".prefill_s" + buckets[i] + ".mlpackage";
            var crc = Sys.command("xcrun coremlc compile " + q(pkg) + " " + q(dir));
            if (crc != 0) {
                Sys.println("nue: coremlc compile failed for " + pkg);
                return crc;
            }
        }
        Sys.println("nue: done — verify with `nue prefill status " + path + "`");
        return 0;
    }

    /** Single-quote a shell argument. Needed because the CLI must build ONE
        command string: `Sys.command(cmd, args)` — supplying an extern's
        optional `?args` — mis-lowers to the wrong arity ("Incorrect number of
        arguments passed to called function"), fails LLVM verification for the
        whole enclosing function, and silences the program. The one-argument
        form is correct, so pass a quoted line. */
    static function q(s:String):String {
        return "'" + StringTools.replace(s, "'", "'\\''") + "'";
    }

    static function usage():Int {
        Sys.println("nue — model tooling");
        Sys.println("  nue info <model.gguf>");
        Sys.println("  nue prefill status <model.gguf> [ctx]");
        Sys.println("  nue prefill author <model.gguf> [bucket ...]");
        return 64;
    }

    static function run(argv:Array<String>):Int {
        if (argv.length < 1) return usage();
        var cmd = argv[0];
        if (cmd == "info" && argv.length >= 2) return info(argv[1]);
        if (cmd == "prefill" && argv.length >= 3) {
            var sub = argv[1];
            if (sub == "status") {
                var ctx = (argv.length >= 4) ? Std.parseInt(argv[3]) : 0;
                return prefillStatus(argv[2], ctx);
            }
            if (sub == "author") {
                var bs:Array<Int> = [];
                for (i in 3...argv.length) {
                    var b = Std.parseInt(argv[i]);
                    if (b != null && b > 0) bs.push(b);
                }
                if (bs.length == 0) bs = [128, 512];
                return prefillAuthor(argv[2], bs);
            }
        }
        return usage();
    }

    public static function main():Void {
        // Flush before exit: Sys.exit does not, and a piped stdout is block
        // buffered.
        var rc = run(Sys.args());
        Sys.stdout().flush();
        Sys.exit(rc);
    }
}
