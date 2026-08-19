package nue.transformer;

/**
 * One conversation's key/value state: a cache per decoder layer.
 *
 * A model's weights are read-only once loaded, so the only thing that stops
 * one loaded model from serving several conversations at once is that the
 * caches used to live on the attention modules. Holding them here instead
 * makes a conversation a value the caller owns: two of them over the same
 * model neither see nor overwrite each other's tokens.
 *
 * Sessions are created by the model, which is the only thing that knows how
 * many layers there are and how each layer's cache was configured.
 */
class KVSession {
    public var caches:Array<KVCache>;

    public function new(caches:Array<KVCache>) {
        this.caches = caches;
    }

    public function layers():Int {
        return caches.length;
    }

    public function cacheFor(layer:Int):KVCache {
        return caches[layer];
    }

    /** Tokens currently held. Every layer advances together, so layer 0
        answers for all of them. */
    public function len():Int {
        return caches.length > 0 ? caches[0].currentLen : 0;
    }

    /** Start a new conversation on the same storage. */
    public function reset():Void {
        for (c in caches) c.reset();
    }

    /** Drop back to an already-written prefix across every layer. */
    public function rewind(len:Int):Void {
        for (c in caches) c.rewind(len);
    }

    /** Release every layer's storage. */
    public function free():Void {
        for (c in caches) c.free();
        caches = [];
    }
}
