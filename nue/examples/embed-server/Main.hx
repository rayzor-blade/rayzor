import nue.arch.BertEmbedder;
import haxe.io.Bytes;
import haxe.io.Eof;
import sys.net.Host;
import sys.net.Socket;

/**
 * nue embed-server — the BERT product surface (Phase 2).
 *
 * Loads a sentence-embedding model ONCE and serves batch embed requests over
 * TCP (sibling of llama-chat-server). A request is newline-separated sentences
 * terminated by the client half-closing its write side; the response is a small
 * text frame: a `count dim` header line, then one L2-normalized embedding per
 * line (space-separated f32). Without `--listen` it runs a smoke batch and
 * prints per-sentence norms, so it is useful without a client.
 *
 *   rayzor run Main.hx -- --model model.gguf --listen 8071 --requests 4
 *   rayzor run Main.hx -- --model model.gguf "sentence one" "sentence two"
 *
 * Model path: --model, a positional *.gguf, NUE_EMBED_MODEL, or GGUF.
 */
class Main {
    static function main() {
        var args = Sys.args();
        var path = modelPathFromEnv();
        var listenPort = 0;
        var requests = 1;
        var sentences:Array<String> = [];

        var i = 0;
        while (i < args.length) {
            var a = args[i];
            if ((a == "--model" || a == "--model-path") && i + 1 < args.length) {
                path = args[i + 1];
                i += 2;
            } else if (a == "--listen" && i + 1 < args.length) {
                listenPort = parsePositiveInt(args[i + 1], listenPort);
                i += 2;
            } else if (a == "--requests" && i + 1 < args.length) {
                requests = parsePositiveInt(args[i + 1], requests);
                i += 2;
            } else if (isEmpty(path) && looksLikeModelPath(a)) {
                path = a;
                i++;
            } else {
                sentences.push(a);
                i++;
            }
        }

        if (isEmpty(path)) {
            usage();
            Sys.exit(1);
        }

        trace("=== nue embed-server ===");
        trace("model:  " + path);
        var loadStart = Sys.time();
        var embedder = new BertEmbedder(path);
        var dim = embedder.dimension();
        trace("[load] done_s=" + fmt(Sys.time() - loadStart) + " dim=" + dim);

        // Smoke mode: no --listen. Embed positional/default sentences and print
        // norms — a self-contained check that the model + pipeline are live.
        if (listenPort <= 0) {
            if (sentences.length == 0) {
                sentences = ["The quick brown fox jumps over the lazy dog.",
                    "A fast auburn animal leaps above a sleepy hound."];
            }
            var t0 = Sys.time();
            for (s in sentences) {
                var e = embedder.embedText(s);
                trace("[embed] dim=" + e.length + " L2=" + fmt(l2(e)) + "  \"" + s + "\"");
            }
            trace("[smoke] " + sentences.length + " sentences in "
                + fmt(Sys.time() - t0) + "s");
            return;
        }

        var server = new Socket();
        server.setFastSend(true);
        server.setTimeout(120.0);
        server.bind(new Host("127.0.0.1"), listenPort);
        server.listen(16);
        trace("[server] listening on 127.0.0.1:" + listenPort + " ready");

        var totalSentences = 0;
        var totalEncode = 0.0;
        for (reqId in 0...requests) {
            var conn = server.accept();
            if (conn == null) continue;
            conn.setFastSend(true);
            conn.setTimeout(120.0);

            var batch = readSentences(conn);
            var t0 = Sys.time();
            var embs:Array<Array<Float>> = [];
            for (s in batch) embs.push(embedder.embedText(s));
            var encode = Sys.time() - t0;

            writeResponse(conn, embs, dim);
            conn.shutdown(false, true);
            conn.close();

            totalSentences += batch.length;
            totalEncode += encode;
            var perS = (encode > 0.0) ? batch.length / encode : 0.0;
            trace("[request-done] id=" + (reqId + 1)
                + " sentences=" + batch.length
                + " encode_s=" + fmt(encode)
                + " sent/s=" + fmt(perS));
        }
        server.close();

        var perS = (totalEncode > 0.0) ? totalSentences / totalEncode : 0.0;
        trace("[server-done] requests=" + requests
            + " sentences=" + totalSentences
            + " encode_s=" + fmt(totalEncode)
            + " sent/s=" + fmt(perS));
        Sys.println("[done] " + totalSentences + " sentences, " + fmt(perS) + " sent/s");
    }

    /** Read the whole request (client half-closes write) → trimmed sentences. */
    static function readSentences(conn:Socket):Array<String> {
        var raw = "";
        try {
            while (true) {
                var chunk = Bytes.alloc(4096);
                var n = conn.input.readBytes(chunk, 0, 4096);
                if (n <= 0) break;
                raw += chunk.sub(0, n).toString();
            }
        } catch (e:Eof) {}
        var out:Array<String> = [];
        for (line in raw.split("\n")) {
            var t = StringTools.trim(line);
            if (t.length > 0) out.push(t);
        }
        return out;
    }

    /** `count dim\n` header, then one space-separated f32 vector per line. */
    static function writeResponse(conn:Socket, embs:Array<Array<Float>>, dim:Int):Void {
        var sb = new StringBuf();
        sb.add(embs.length);
        sb.add(" ");
        sb.add(dim);
        sb.add("\n");
        for (e in embs) {
            for (j in 0...dim) {
                if (j > 0) sb.add(" ");
                sb.add(e[j]);
            }
            sb.add("\n");
        }
        conn.output.writeString(sb.toString());
        conn.output.flush();
    }

    static function l2(e:Array<Float>):Float {
        var s = 0.0;
        for (v in e) s += v * v;
        return Math.sqrt(s);
    }

    static function usage():Void {
        Sys.println("usage: rayzor run Main.hx -- [--model model.gguf] [--listen PORT] [--requests N] [sentence...]");
        Sys.println("       Model: --model, a positional *.gguf, NUE_EMBED_MODEL, or GGUF.");
        Sys.println("       Request = newline-separated sentences, client half-closes write.");
    }

    static function modelPathFromEnv():String {
        var path = Sys.getEnvOr("NUE_EMBED_MODEL", "RAYZOR_EMBED_MODEL");
        if (!isEmpty(path)) return path;
        path = Sys.getEnv("GGUF");
        if (!isEmpty(path)) return path;
        return null;
    }

    static function isEmpty(s:String):Bool {
        return s == null || s.length == 0;
    }

    static function looksLikeModelPath(s:String):Bool {
        return s.indexOf(".gguf") >= 0;
    }

    static function parsePositiveInt(s:String, fallback:Int):Int {
        var parsed = Std.parseInt(s);
        if (parsed == null) return fallback;
        var v = parsed + 0;
        return (v > 0) ? v : fallback;
    }

    public static inline function fmt(x:Float):String {
        return Std.string(Math.round(x * 1000) / 1000);
    }
}
