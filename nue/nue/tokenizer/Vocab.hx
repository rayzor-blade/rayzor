package nue.tokenizer;

/**
 * Bidirectional vocabulary: token string ↔ integer ID. Built once at
 * tokenizer construction time (typically by reading the model file's
 * embedded vocab table) and immutable thereafter.
 *
 * **Implementation choice (current):** parallel arrays (`tokens`,
 * `scores`) + a sorted-by-token side index for O(log N) lookup via
 * binary search. The pure-Haxe `Map<String, V>.get` path in the
 * current rayzor stdlib doesn't return values correctly, and we
 * don't want to wait on that fix — binary search is fast enough for
 * tokenisation (sub-1% of inference time) and avoids the dependency.
 *
 * `scores` is optional — SentencePiece models carry per-token
 * log-probabilities used for tie-breaking during BPE merge selection,
 * while plain tiktoken-style BPE leaves them empty.
 */
class Vocab {
    public var tokens:Array<String>;
    public var scores:Array<Float>;
    /** Sorted-by-string index: `sortedIdx[k]` is the ID whose token
        is the k-th in lexicographic order. Rebuilt lazily. */
    private var sortedIdx:Array<Int>;
    private var sortedDirty:Bool;

    public function new() {
        this.tokens = [];
        this.scores = [];
        this.sortedIdx = [];
        this.sortedDirty = false;
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
        sortedDirty = true;
        return id;
    }

    public inline function size():Int {
        return tokens.length;
    }

    public inline function get(id:Int):String {
        return tokens[id];
    }

    /**
     * Token → ID. Returns -1 when not found. Binary search after a
     * one-shot O(N log N) sort the first time a lookup happens after
     * any `add()`.
     */
    public function lookup(token:String):Int {
        if (sortedDirty) rebuildIndex();
        var lo = 0;
        var hi = sortedIdx.length - 1;
        while (lo <= hi) {
            var mid = (lo + hi) >> 1;
            var id = sortedIdx[mid];
            var cmp = compare(tokens[id], token);
            if (cmp == 0) return id;
            if (cmp < 0) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    private function rebuildIndex():Void {
        sortedIdx = [];
        for (i in 0...tokens.length) sortedIdx.push(i);
        // Insertion sort — fine for the typical "fill once, lookup many" pattern.
        // For 128k vocabs at construction we'd want something better; revisit
        // when a real tokenizer test motivates it.
        for (i in 1...sortedIdx.length) {
            var x = sortedIdx[i];
            var j = i;
            while (j > 0 && compare(tokens[sortedIdx[j - 1]], tokens[x]) > 0) {
                sortedIdx[j] = sortedIdx[j - 1];
                j--;
            }
            sortedIdx[j] = x;
        }
        sortedDirty = false;
    }

    /** Stable lexicographic string compare returning -1/0/1. */
    private static function compare(a:String, b:String):Int {
        var la = a.length;
        var lb = b.length;
        var n = la < lb ? la : lb;
        for (i in 0...n) {
            var ca = a.charCodeAt(i);
            var cb = b.charCodeAt(i);
            if (ca < cb) return -1;
            if (ca > cb) return 1;
        }
        if (la < lb) return -1;
        if (la > lb) return 1;
        return 0;
    }
}
