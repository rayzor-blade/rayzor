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

    /** Token → ID. Returns -1 when not found. */
    public function lookup(token:String):Int {
        var id = idOf.get(token);
        return (id == null) ? -1 : id;
    }
}
