import nue.loader.GGUFLoader;
import nue.sampling.GenerationLoop;
import nue.sampling.Sampler;
import nue.tokenizer.Tokenizer;
import nue.CausalLanguageModel;
import rayzor.ds.Tensor;
import haxe.io.Bytes;
import sys.net.Host;
import sys.net.Socket;

class LocalTempSampler implements Sampler {
    public var temperature:Float;
    public var repetitionPenalty:Float;
    public var topK:Int;
    private var state:Int;
    private var recent:Array<Int>;
    private var recentWrite:Int;
    private var topKLogits:Array<Float>;
    private var topKIds:Array<Int>;
    private static inline var RECENT_CAP:Int = 64;

    public function new(temperature:Float, repetitionPenalty:Float, topK:Int, seed:Int) {
        this.temperature = temperature;
        this.repetitionPenalty = repetitionPenalty;
        this.topK = topK;
        this.state = seed;
        this.recent = [for (_ in 0...RECENT_CAP) -1];
        this.recentWrite = 0;
        var k = (topK > 0) ? topK : 1;
        this.topKLogits = [for (_ in 0...k) 0.0];
        this.topKIds = [for (_ in 0...k) -1];
    }

    public function sample(logits:Tensor):Int {
        var shape = logits.shape();
        var n = shape[shape.length - 1];
        var t = (temperature <= 0.0) ? 0.00000001 : temperature;
        var penalize = repetitionPenalty > 1.0;
        var rp = repetitionPenalty;
        var k = (topK > 0 && topK < n) ? topK : n;

        var penaltyArg = penalize ? rp : 1.0;
        var sz = logits.topkScan(topKLogits, topKIds, k, recent, penaltyArg);
        if (sz < 0) {
            sz = topKScanFallback(logits, n, k, penalize, rp);
        }

        var maxLogit = topKLogits[0];
        var total = 0.0;
        for (i in 0...sz) {
            total += Math.exp((topKLogits[i] - maxLogit) / t);
        }

        var r = nextFloat() * total;
        var acc = 0.0;
        for (i in 0...sz) {
            acc += Math.exp((topKLogits[i] - maxLogit) / t);
            if (r <= acc) {
                var id = topKIds[i];
                pushRecent(id);
                return id;
            }
        }
        var id = topKIds[sz - 1];
        pushRecent(id);
        return id;
    }

    private function topKScanFallback(
        logits:Tensor, n:Int, k:Int, penalize:Bool, rp:Float
    ):Int {
        var sz = 0;
        for (i in 0...n) {
            var lg = adjusted(logits.getFlat(i), i, penalize, rp);
            if (sz < k) {
                var pos = sz;
                while (pos > 0 && topKLogits[pos - 1] < lg) {
                    topKLogits[pos] = topKLogits[pos - 1];
                    topKIds[pos] = topKIds[pos - 1];
                    pos--;
                }
                topKLogits[pos] = lg;
                topKIds[pos] = i;
                sz++;
            } else if (lg > topKLogits[k - 1]) {
                var pos = k - 1;
                while (pos > 0 && topKLogits[pos - 1] < lg) {
                    topKLogits[pos] = topKLogits[pos - 1];
                    topKIds[pos] = topKIds[pos - 1];
                    pos--;
                }
                topKLogits[pos] = lg;
                topKIds[pos] = i;
            }
        }
        return sz;
    }

    private inline function adjusted(lg:Float, id:Int, penalize:Bool, rp:Float):Float {
        if (!penalize) return lg;
        if (!isRecent(id)) return lg;
        return (lg > 0.0) ? lg / rp : lg * rp;
    }

    private function isRecent(id:Int):Bool {
        for (k in 0...RECENT_CAP) {
            if (recent[k] == id) return true;
        }
        return false;
    }

    private function pushRecent(id:Int):Void {
        recent[recentWrite] = id;
        recentWrite = (recentWrite + 1) % RECENT_CAP;
    }

    private function nextFloat():Float {
        state = (state * 1664525 + 1013904223) & 0x7FFFFFFF;
        return state / 2147483648.0;
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
        var stream = Sys.getEnv("RAYZOR_SERVER_STREAM") != null;
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

        var envPrompts = Sys.getEnv("RAYZOR_SERVER_PROMPTS");
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

        var model = cast(loaded.model, CausalLanguageModel);
        var tok = loaded.tokenizer;
        var meta = loaded.metadata;
        trace("[meta] " + meta.architecture + " hidden=" + meta.hiddenSize
            + " layers=" + meta.numLayers + " heads=" + meta.numHeads
            + "/" + meta.numKvHeads + " ffn=" + meta.intermediateSize
            + " vocab=" + meta.vocabSize + " ctx=" + meta.maxSeqLen);

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

            var sampler:Sampler = (temperature > 0.0)
                ? new LocalTempSampler(temperature, 1.15, 50, 42 + reqNo)
                : new LocalTempSampler(1e-4, 1.0, 1, 42 + reqNo);
            var loop = new GenerationLoop(model, tok, sampler, eos, maxNew);

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
            if (stream) {
                output = loop.generate(modelPrompt, function(_id:Int, delta:String):Bool {
                    Sys.print(delta);
                    var out = socketOutput[0];
                    if (out != null) {
                        out.writeString(delta);
                        out.flush();
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
            if (Sys.getEnv("RAYZOR_LLAMA_DUMP_OUTPUT") != null) {
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
    }

    static function usage():Void {
        Sys.println("usage: rayzor run Main.hx -- [--model model.gguf] [--max N] [--ctx N] [--temp F] [--requests N] [--listen PORT] [--stream] [prompt...]");
        Sys.println("       Server model config: RAYZOR_SERVER_MODEL=/path/model.gguf or GGUF=/path/model.gguf.");
        Sys.println("       Request prompts: positional args, TCP request bytes, or RAYZOR_SERVER_PROMPTS='p1|||p2'.");
    }

    static function modelPathFromEnv():String {
        var path = Sys.getEnv("RAYZOR_SERVER_MODEL");
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
