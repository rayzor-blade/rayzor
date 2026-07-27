import nue.loader.GGUFLoader;
import nue.sampling.GenerationLoop;
import nue.sampling.LocalTempSampler;
import nue.tokenizer.Tokenizer;
import nue.tokenizer.BPETokenizer;
import nue.arch.LlamaModel;
import rayzor.ds.Tensor;
import haxe.io.Bytes;
import sys.net.Host;
import sys.net.Socket;
import rayzor.concurrent.Arc;
import rayzor.concurrent.Channel;
import rayzor.concurrent.Thread;

@:derive([Send])
class StreamMsg {
    public var text:String;
    public var done:Bool;
    public function new(text:String, done:Bool) {
        this.text = text;
        this.done = done;
    }
}

class Main {
    static inline var DEFAULT_MAX_TOKENS = 128;
    static inline var DEFAULT_CTX = 4096;
    static inline var DEFAULT_TEMP = 0.0;
    static inline var DEFAULT_REQUESTS = 3;
    static inline var DEFAULT_PROMPT =
        "Explain voronoi regions and their connection to delaunay computation.";

    static function main() {
        var args = Sys.args();
        var path = modelPathFromEnv();
        var maxNew = DEFAULT_MAX_TOKENS;
        var ctx = DEFAULT_CTX;
        var temperature = DEFAULT_TEMP;
        var requests = DEFAULT_REQUESTS;
        var listenPort = 0;
        var stream = Sys.getEnvOr("NUE_SERVER_STREAM", "RAYZOR_SERVER_STREAM") != null;
        var prompts:Array<String> = [];

        var i = 0;
        while (i < args.length) {
            var arg = args[i];
            if ((arg == "--model" || arg == "--model-path") && i + 1 < args.length) {
                path = args[i + 1];
                i += 2;
            } else if (arg == "--max" && i + 1 < args.length) {
                maxNew = parsePositiveInt(args[i + 1], maxNew);
                i += 2;
            } else if (arg == "--ctx" && i + 1 < args.length) {
                ctx = parsePositiveInt(args[i + 1], ctx);
                i += 2;
            } else if (arg == "--temp" && i + 1 < args.length) {
                temperature = parseNonNegativeFloat(args[i + 1], temperature);
                i += 2;
            } else if (arg == "--requests" && i + 1 < args.length) {
                requests = parsePositiveInt(args[i + 1], requests);
                i += 2;
            } else if (arg == "--listen" && i + 1 < args.length) {
                listenPort = parsePositiveInt(args[i + 1], listenPort);
                i += 2;
            } else if (arg == "--stream") {
                stream = true;
                i++;
            } else if (isEmpty(path) && looksLikeModelPath(arg)) {
                path = arg;
                i++;
            } else {
                prompts.push(arg);
                i++;
            }
        }

        if (isEmpty(path)) {
            usage();
            Sys.exit(1);
        }

        var envPrompts = Sys.getEnvOr("NUE_SERVER_PROMPTS", "RAYZOR_SERVER_PROMPTS");
        if (prompts.length == 0 && envPrompts != null && envPrompts.length > 0) {
            prompts = envPrompts.split("|||");
        }
        if (prompts.length == 0) {
            prompts.push(DEFAULT_PROMPT);
        }

        trace("=== nue llama-chat-server ===");
        trace("model:    " + path);
        trace("requests: " + requests);
        trace("max:      " + maxNew + " tokens/request");
        trace("ctx:      " + ctx + " tokens");
        trace("temp:     " + temperature);
        trace("stream:   " + stream);
        if (listenPort > 0) trace("listen:   127.0.0.1:" + listenPort);
        trace("");

        trace("[server] loading GGUF once");
        var loadStarted = Sys.time();
        var loader = new GGUFLoader();
        var loaded = loader.loadWithTokenizer(path, ctx);
        trace("[server] load_s=" + fmt(Sys.time() - loadStarted));

        var llama = cast(loaded.model, LlamaModel);
        var profilePool = Sys.getEnvOr("NUE_PROFILE_POOL", "RAYZOR_PROFILE_POOL") != null;
        // Tokenizer arrives as the format-agnostic interface; the generation
        // loop takes the concrete BPETokenizer (checked cast, as with the model).
        var tok = cast(loaded.tokenizer, BPETokenizer);
        var meta = loaded.metadata;
        trace("[meta] " + meta.architecture + " hidden=" + meta.hiddenSize
            + " layers=" + meta.numLayers + " heads=" + meta.numHeads
            + "/" + meta.numKvHeads + " ffn=" + meta.intermediateSize
            + " vocab=" + meta.vocabSize + " ctx=" + meta.maxSeqLen);
        if (profilePool && llama.spinPool != null) {
            trace("[pool] workers=" + llama.spinPool.workers());
        }

        var eos = tok.specialId("<|eot_id|>");
        if (eos < 0) eos = tok.specialId("<|end_of_text|>");
        var vocab = tok.vocabSize();
        trace("[tok] vocab=" + vocab + " eos=" + eos);
        trace("[server] ready");

        var server:Socket = null;
        if (listenPort > 0) {
            server = new Socket();
            server.setFastSend(true);
            server.setTimeout(120.0);
            server.bind(new Host("127.0.0.1"), listenPort);
            server.listen(16);
            trace("[server] listening on 127.0.0.1:" + listenPort);
        }

        var totalTokens = 0;
        var totalDecode = 0.0;
        var wallStart = Sys.time();
        for (requestId in 0...requests) {
            var conn:Socket = null;
            var prompt = prompts[requestId % prompts.length];
            var reqNo = requestId + 1;
            if (server != null) {
                trace("[server] waiting for request " + reqNo + "/" + requests);
                conn = server.accept();
                if (conn == null) {
                    trace("[server] accept returned null");
                    continue;
                }
                conn.setFastSend(true);
                conn.setTimeout(120.0);
                prompt = readSocketRequest(conn);
                trace("[server] accepted request " + reqNo
                    + " bytes=" + prompt.length);
                if (prompt.length == 0) prompt = DEFAULT_PROMPT;
            }

            var sampler:LocalTempSampler = (temperature > 0.0)
                ? new LocalTempSampler(temperature, 1.15, 50, 42 + reqNo)
                : new LocalTempSampler(1e-4, 1.0, 1, 42 + reqNo);
            // Concrete LlamaModel + shared LocalTempSampler: the hot path must
            // not cross an interface boundary (open x-module iface dispatch bug).
            var loop = new GenerationLoop(llama, tok, sampler, eos, maxNew);

            var promptLen = prompt.length;
            var startHdr = tok.specialId("<|start_header_id|>");
            var modelPrompt:String = prompt;
            if (startHdr >= 0) {
                modelPrompt =
                    "<|begin_of_text|>"
                    + "<|start_header_id|>user<|end_header_id|>\n\n"
                    + prompt
                    + "<|eot_id|>"
                    + "<|start_header_id|>assistant<|end_header_id|>\n\n";
            }

            trace("[request] id=" + reqNo + " prompt_len=" + promptLen
                + " model_prompt_len=" + modelPrompt.length);

            var nTokens = [0];
            var socketOutput = [conn != null ? conn.output : null];
            var started = Sys.time();
            var output:String;
            var out = socketOutput[0];
            var writerOff = Sys.getEnvOr("NUE_STREAM_WRITER", "RAYZOR_STREAM_WRITER") == "0";
            if (stream && out != null && !writerOff) {
                // Async streaming: decode sends each token delta to an unbounded
                // channel (non-blocking send), and a writer thread owns the
                // socket and coalesces writeString/flush OFF the decode thread.
                // The per-token TCP flush no longer stalls decode, so STREAM=1
                // tracks STREAM=0 (~90 tok/s) instead of contending with it.
                var chArc = new Arc(new Channel(0));
                var writerCh = chArc.clone();
                var writer:Thread<Int> = Thread.spawn(function():Int {
                    while (true) {
                        var m:StreamMsg = writerCh.get().receive();
                        if (m == null || m.done) break;
                        var s = m.text;
                        var n:StreamMsg = writerCh.get().tryReceive();
                        while (n != null) {
                            if (n.done) {
                                out.writeString(s);
                                out.flush();
                                return 0;
                            }
                            s += n.text;
                            n = writerCh.get().tryReceive();
                        }
                        out.writeString(s);
                        out.flush();
                    }
                    return 0;
                });
                // Producer-side coalescing: per-token work is a cheap array
                // push; the channel is crossed only per BATCH, so decode isn't
                // taxed per token (the earlier per-token send was as costly as
                // the per-token flush it replaced). The writer still does the
                // socket I/O off the decode thread.
                var pending:Array<Array<String>> = [[]];
                output = loop.generate(modelPrompt, function(_id:Int, delta:String):Bool {
                    pending[0].push(delta);
                    if (pending[0].length >= 8) {
                        chArc.get().send(new StreamMsg(pending[0].join(""), false));
                        pending[0] = [];
                    }
                    nTokens[0] = nTokens[0] + 1;
                    return true;
                });
                if (pending[0].length > 0) {
                    chArc.get().send(new StreamMsg(pending[0].join(""), false));
                }
                chArc.get().send(new StreamMsg("", true));
                writer.join();
            } else if (stream) {
                // Sync fallback: writer disabled (NUE_STREAM_WRITER=0) or no socket.
                output = loop.generate(modelPrompt, function(_id:Int, delta:String):Bool {
                    var o = socketOutput[0];
                    if (o != null) {
                        o.writeString(delta);
                        o.flush();
                    }
                    nTokens[0] = nTokens[0] + 1;
                    return true;
                });
            } else {
                output = loop.generate(modelPrompt, function(_id:Int, _partial:String):Bool {
                    nTokens[0] = nTokens[0] + 1;
                    return true;
                });
            }
            var seconds = Sys.time() - started;
            if (stream) Sys.println("");
            if (conn != null) {
                if (!stream) {
                    conn.output.writeString(output);
                    conn.output.flush();
                }
                conn.shutdown(false, true);
                conn.close();
            }
            var tokens = nTokens[0];
            trace("[request-done] id=" + reqNo
                + " tokens=" + tokens
                + " seconds=" + fmt(seconds)
                + " tok/s=" + fmt(tokens / seconds)
                + " output_len=" + output.length);
            if (profilePool && llama.spinPool != null) {
                trace("[profile-pool] request=" + reqNo
                    + " " + llama.spinPool.profReport());
            }
            // Effective plan, once per request under NUE_DUMP_Q4_GATES=1 —
            // the server is where stale gates actually cost throughput.
            nue.Q4Matmul.dumpPlan();
            if (Sys.getEnvOr("NUE_LLAMA_DUMP_OUTPUT", "RAYZOR_LLAMA_DUMP_OUTPUT") != null) {
                trace("[output] request=" + reqNo + " " + output);
            }
            totalTokens += tokens;
            totalDecode += seconds;
        }
        if (server != null) server.close();

        var wall = Sys.time() - wallStart;
        trace("[server-done] requests=" + requests
            + " decode_s=" + fmt(totalDecode)
            + " wall_s=" + fmt(wall)
            + " tokens=" + totalTokens
            + " tok/s=" + fmt(totalTokens / totalDecode));
        Sys.println("[done] " + totalTokens + " tokens in " + fmt(totalDecode)
            + "s (" + fmt(totalTokens / totalDecode) + " tok/s)");
        llama.shutdownPool();
    }

    static function usage():Void {
        Sys.println("usage: rayzor run Main.hx -- [--model model.gguf] [--max N] [--ctx N] [--temp F] [--requests N] [--listen PORT] [--stream] [prompt...]");
        Sys.println("       Server model config: RAYZOR_SERVER_MODEL=/path/model.gguf or GGUF=/path/model.gguf.");
        Sys.println("       Request prompts: positional args, TCP request bytes, or RAYZOR_SERVER_PROMPTS='p1|||p2'.");
    }

    static function modelPathFromEnv():String {
        var path = Sys.getEnvOr("NUE_SERVER_MODEL", "RAYZOR_SERVER_MODEL");
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

    static function parseNonNegativeFloat(s:String, fallback:Float):Float {
        var parsed = Std.parseFloat(s);
        if (parsed != parsed || parsed < 0.0) return fallback;
        return parsed;
    }

    static function readSocketRequest(conn:Socket):String {
        var result = "";
        while (true) {
            var chunk = Bytes.alloc(4096);
            var n = conn.input.readBytes(chunk, 0, 4096);
            if (n <= 0) break;
            result += chunk.sub(0, n).toString();
            if (n < 4096) break;
        }
        return result;
    }

    public static inline function fmt(x:Float):String {
        return Std.string(Math.round(x * 1000) / 1000);
    }
}
