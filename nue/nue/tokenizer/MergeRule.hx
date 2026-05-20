package nue.tokenizer;

/**
 * A single BPE merge rule: `left + right → merged`. Stored as a
 * named class rather than an anonymous `{left, right, merged}`
 * record because array-of-anonymous-object dispatch trips a
 * Cranelift codegen edge case in the current JIT — see
 * bugs_known.md.
 *
 * `key = left + right` is precomputed so the encode hot loop can
 * compare one string per pair instead of two.
 */
class MergeRule {
    public var left:String;
    public var right:String;
    public var merged:String;
    public var key:String;

    public function new(left:String, right:String, merged:String) {
        this.left = left;
        this.right = right;
        this.merged = merged;
        this.key = left + right;
    }
}
