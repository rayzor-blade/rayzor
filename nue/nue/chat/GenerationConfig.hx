package nue.chat;

/**
 * Sampling and length controls for a generation request.
 *
 * Grouped into one object so a caller can build a preset once and reuse or
 * clone it per request, rather than threading half a dozen loose arguments.
 * Defaults target small instruct models; `repetitionPenalty` at 1.3 is the
 * load-bearing one — lower values let a 1B model fall into sentence loops on
 * long generations.
 */
class GenerationConfig {
    /** Hard cap on generated tokens (0 = until a stop token / context end). */
    public var maxNewTokens:Int;
    /** 0.0 = greedy/deterministic; higher = more random. */
    public var temperature:Float;
    /** >1.0 demotes recently-emitted tokens before sampling to break loops. */
    public var repetitionPenalty:Float;
    /** Sample from the top-K logits (ignored at temperature 0). */
    public var topK:Int;
    /** RNG seed for reproducible sampling. */
    public var seed:Int;
    /** Route decoding through the n-gram speculative loop. */
    public var useSpeculative:Bool;

    public function new() {
        this.maxNewTokens = 512;
        this.temperature = 0.7;
        this.repetitionPenalty = 1.3;
        this.topK = 50;
        this.seed = 42;
        this.useSpeculative = false;
    }

    /** Deterministic, reproducible output (greedy) — penalty still applied. */
    public static function greedy():GenerationConfig {
        var c = new GenerationConfig();
        c.temperature = 0.0;
        return c;
    }

    /** Balanced conversational default. */
    public static function balanced():GenerationConfig {
        return new GenerationConfig();
    }

    /** Higher-variance, more exploratory output. */
    public static function creative():GenerationConfig {
        var c = new GenerationConfig();
        c.temperature = 0.9;
        c.topK = 80;
        return c;
    }

    /** An independent copy, so per-request tweaks don't mutate the original. */
    public function copy():GenerationConfig {
        var c = new GenerationConfig();
        c.maxNewTokens = maxNewTokens;
        c.temperature = temperature;
        c.repetitionPenalty = repetitionPenalty;
        c.topK = topK;
        c.seed = seed;
        c.useSpeculative = useSpeculative;
        return c;
    }
}
