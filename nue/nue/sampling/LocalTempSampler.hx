package nue.sampling;

import rayzor.ds.Tensor;

/**
 * Top-K + temperature + repetition-penalty sampler used by llama-chat.
 *
 * This is intentionally concrete so the decode loop does not pay an
 * interface-dispatch / late-return-type penalty in MIR. It keeps the exact
 * sampling logic that previously lived inside `Main.hx`.
 */
class LocalTempSampler implements Sampler {
    public var temperature:Float;
    public var repetitionPenalty:Float;
    public var topK:Int;
    private var state:Int;
    private var recent:Array<Int>;
    private var recentWrite:Int;
    private var topKLogits:Array<Float>;
    private var topKIds:Array<Int>;
    private var survIdx:Array<Int>;
    private var seen:Array<Int>;
    private var ngPrefix:Array<Int>;
    private var ngFilled:Int;
    private static inline var NG_N:Int = 8;
    private static inline var RECENT_CAP:Int = 64;
    private static inline var NG_TABLE:Int = 1 << 16;
    private static inline var NG_MASK:Int = (1 << 16) - 1;

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
        this.survIdx = [for (_ in 0...k) 0];
        this.seen = [for (_ in 0...NG_TABLE) -1];
        this.ngPrefix = [for (_ in 0...(NG_N - 1)) -1];
        this.ngFilled = 0;
    }

    private function ngHashCand(cand:Int):Int {
        var h = 0;
        for (i in 0...(NG_N - 1)) {
            h = (h ^ ngPrefix[i]) * 1000003;
        }
        h = (h ^ cand) * 1000003;
        return h & 0x7FFFFFFF;
    }

    private function ngContainsCand(cand:Int):Bool {
        var h = ngHashCand(cand);
        var slot = h & NG_MASK;
        for (_ in 0...NG_TABLE) {
            var v = seen[slot];
            if (v == -1) return false;
            if (v == h) return true;
            slot = (slot + 1) & NG_MASK;
        }
        return false;
    }

    private function ngPush(cand:Int):Void {
        if (ngFilled >= NG_N - 1) {
            var h = ngHashCand(cand);
            var slot = h & NG_MASK;
            for (_ in 0...NG_TABLE) {
                var v = seen[slot];
                if (v == -1) {
                    seen[slot] = h;
                    break;
                }
                if (v == h) break;
                slot = (slot + 1) & NG_MASK;
            }
        }
        for (i in 0...(NG_N - 2)) {
            ngPrefix[i] = ngPrefix[i + 1];
        }
        ngPrefix[NG_N - 2] = cand;
        if (ngFilled < NG_N - 1) ngFilled++;
    }

    static var _dumpTopkGate:Int = 0;
    static var _dumpStep:Int = 0;

    static inline function dumpTopk():Bool {
        if (_dumpTopkGate == 0) {
            var v = Sys.getEnvOr("NUE_DUMP_TOPK", "RAYZOR_DUMP_TOPK");
            _dumpTopkGate = (v != null && v != "0" && v != "" && v != "false") ? 1 : 2;
        }
        return _dumpTopkGate == 1;
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

        if (dumpTopk()) {
            var line = "[topk] step=" + _dumpStep;
            var m = sz < 5 ? sz : 5;
            for (i in 0...m) line += " " + topKIds[i] + ":" + topKLogits[i];
            Sys.println(line);
            _dumpStep++;
        }

        var nSurv = 0;
        if (ngFilled >= NG_N - 1) {
            for (i in 0...sz) {
                if (!ngContainsCand(topKIds[i])) {
                    survIdx[nSurv] = i;
                    nSurv++;
                }
            }
        }
        if (nSurv == 0) {
            for (i in 0...sz) survIdx[i] = i;
            nSurv = sz;
        }

        var maxLogit = topKLogits[survIdx[0]];
        var total = 0.0;
        for (s in 0...nSurv) {
            total += Math.exp((topKLogits[survIdx[s]] - maxLogit) / t);
        }

        var r = nextFloat() * total;
        var acc = 0.0;
        var chosen = survIdx[nSurv - 1];
        for (s in 0...nSurv) {
            acc += Math.exp((topKLogits[survIdx[s]] - maxLogit) / t);
            if (r <= acc) {
                chosen = survIdx[s];
                break;
            }
        }
        var id = topKIds[chosen];
        ngPush(id);
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
