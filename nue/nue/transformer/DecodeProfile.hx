package nue.transformer;

/**
 * Where a decode step actually goes, per bucket.
 *
 * `GenerationLoop` times the whole transformer as one `fwd` bucket, and
 * `NUE_PROFILE_ATTN` split attention out of it — which showed the attention
 * scan is 6-7% of Mistral-7B decode at chat lengths, leaving ~72% in a single
 * undifferentiated remainder. This splits that remainder so the next lever is
 * chosen against a measurement instead of an assumed cost model.
 *
 * Buckets are per-layer unless noted:
 *   0 norm    RMSNorm x2 (attn + ffn)
 *   1 attn    the whole GQAttention.forward (already split by NUE_PROFILE_ATTN)
 *   2 ffn     SwiGLU: gate/up/down matmuls + activation
 *   3 resid   the two residual addInto
 *   4 lmhead  final vocab projection — ONCE per token, not per layer
 *
 * Enabled by `NUE_PROFILE_DECODE_SPLIT=1`. Primitive params only and a no-arg
 * dump: a cross-module type on either trips the x-module static-resolution
 * cluster documented on `Q4Matmul.dumpPlan`.
 */
class DecodeProfile {
    static var _sNorm:Float = 0.0;
    static var _sAttn:Float = 0.0;
    static var _sFfn:Float = 0.0;
    static var _sResid:Float = 0.0;
    static var _sLmHead:Float = 0.0;
    static var _nBlocks:Int = 0;
    static var _nTokens:Int = 0;
    static var _on:Int = 0;

    public static function enabled():Bool {
        if (_on == 0) {
            var v = Sys.getEnvOr("NUE_PROFILE_DECODE_SPLIT", "RAYZOR_PROFILE_DECODE_SPLIT");
            _on = (v != null && v != "0" && v != "" && v != "false") ? 1 : 2;
        }
        return _on == 1;
    }

    /** Timestamp when profiling is on, else 0. Keeps `Sys.time()` out of the
        hot path when off, and keeps callers to a ternary rather than
        `if (c) f = expr`, which boxes a loop-carried Float per iteration. */
    public static function now():Float {
        return enabled() ? Sys.time() : 0.0;
    }

    /** One layer's split. Unconditional adds — see `now()`. */
    public static function noteBlock(norm:Float, attn:Float, ffn:Float, resid:Float):Void {
        if (!enabled()) return;
        _sNorm = _sNorm + norm;
        _sAttn = _sAttn + attn;
        _sFfn = _sFfn + ffn;
        _sResid = _sResid + resid;
        _nBlocks = _nBlocks + 1;
    }

    /** One token's final vocab projection. */
    public static function noteLmHead(seconds:Float):Void {
        if (!enabled()) return;
        _sLmHead = _sLmHead + seconds;
        _nTokens = _nTokens + 1;
    }

    static function pct(part:Float, total:Float):String {
        if (total <= 0.0) return "0.0%";
        return Std.string(Math.round((part / total) * 1000.0) / 10.0) + "%";
    }

    static function secs(v:Float):String {
        return Std.string(Math.round(v * 1000.0) / 1000.0) + "s";
    }

    public static function dump():Void {
        if (!enabled()) return;
        var total = _sNorm + _sAttn + _sFfn + _sResid + _sLmHead;
        if (total <= 0.0) {
            Sys.println("[decode-split] no block forward observed");
            return;
        }
        Sys.println("[decode-split] blocks=" + _nBlocks + " tokens=" + _nTokens
            + "  ffn=" + secs(_sFfn) + " (" + pct(_sFfn, total) + ")"
            + "  attn=" + secs(_sAttn) + " (" + pct(_sAttn, total) + ")"
            + "  lmhead=" + secs(_sLmHead) + " (" + pct(_sLmHead, total) + ")"
            + "  norm=" + secs(_sNorm) + " (" + pct(_sNorm, total) + ")"
            + "  resid=" + secs(_sResid) + " (" + pct(_sResid, total) + ")");
    }
}
