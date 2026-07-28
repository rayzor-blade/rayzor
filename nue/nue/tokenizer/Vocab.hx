package nue.tokenizer;

import haxe.ds.StringMap;

/**
 * Bidirectional vocabulary: token string ↔ integer ID. Built once at
 * tokenizer construction time (typically by reading the model file's
 * embedded vocab table) and immutable thereafter.
 *
 * Backed by parallel arrays + a `StringMap<Int>` for O(1) `lookup`.
 * The arrays preserve insertion order so `tokens[id]` works in either
 * direction; the map accelerates string-to-id which is the hot path
 * during BPE merge resolution.
 *
 * `scores` is optional — SentencePiece models carry per-token
 * log-probabilities used for tie-breaking during BPE merge selection,
 * while plain tiktoken-style BPE leaves them empty.
 */
class Vocab {
    public var tokens:Array<String>;
    public var scores:Array<Float>;
    /** Hot-path token → ID lookup. O(1) on average. */
    public var idOf:StringMap<Int>;

    public function new() {
        this.tokens = [];
        this.scores = [];
        this.idOf = new StringMap<Int>();
    }

    /** Append a token. Returns its assigned ID. */
    public function add(token:String):Int {
        return addWithScore(token, 0.0);
    }

    /** Append a token with explicit score (SentencePiece BPE merge tie-breaker). */
    public function addWithScore(token:String, score:Float):Int {
        var id = tokens.length;
        tokens.push(token);
        scores.push(score);
        idOf.set(token, id);
        return id;
    }

    public inline function size():Int {
        return tokens.length;
    }

    public inline function get(id:Int):String {
        return tokens[id];
    }

    /** Token → ID. Returns -1 when not found.
        Uses an explicit `exists` check because the runtime's
        `StringMap<Int>.get` returns raw `0u64` for missing keys
        rather than boxed null — so `get(...) == null` would be
        false for missing keys and miscompare against a real 0 entry
        (see `bugs_stringmap_null_get.md`). */
    /** Unigram score for `id`, or 0 when absent.

        An ACCESSOR, not a field read: `vocab.scores[id]` from another module
        goes through the cross-module instance-field path, which this tree has
        a documented history of resolving to garbage. Method dispatch does not. */
    public function scoreOf(id:Int):Float {
        if (id < 0 || id >= scores.length) return 0.0;
        return scores[id];
    }

    /** Token text for `id`, or "" when out of range. Same reasoning as
        `scoreOf` — callers in other modules must not index `tokens` directly. */
    public function tokenAt(id:Int):String {
        if (id < 0 || id >= tokens.length) return "";
        return tokens[id];
    }

    public function lookup(token:String):Int {
        if (!idOf.exists(token)) return -1;
        return idOf.get(token);
    }
}
