// Regression: a generic class whose constructor has untyped, self-referentially
// typed params (lowered to *void) plus an in-constructor method call used to leave
// the constructor un-monomorphized — the generic template reached codegen still
// carrying type_params, got a forward-ref trap stub, and the `new` call SIGILL'd
// (udf #0xc11f / exit 132). The monomorphizer's argument-based inference can't
// recover K/V from *void params, so the `new Foo<Int,Int>(...)` call site must
// carry the explicit type_args. Fixed in hir_to_mir New lowering
// (build_call_direct_with_type_args). This is the shape of haxe.ds.TreeNode
// (self-referential `left`/`right` + `get_height()` called inside `new`).

class TreeNode<K, V> {
    public var left:TreeNode<K, V>;
    public var right:TreeNode<K, V>;
    public var key:K;
    public var value:V;
    var _height:Int;

    // Untyped params (l, k, v, r) — inferred, lower to *void. Default last param.
    public function new(l, k, v, r, h = -1) {
        left = l; key = k; value = v; right = r;
        if (h == -1) {
            // Method call on a self-referentially typed field inside the ctor —
            // the trigger that previously broke monomorphization.
            var lh = left.get_height();
            var rh = right.get_height();
            _height = (lh > rh ? lh : rh) + 1;
        } else {
            _height = h;
        }
    }

    public function get_height():Int {
        if (this == null) return 0;
        return _height;
    }
}

class Main {
    static function main() {
        // Leaf node: left/right null, get_height() guards this==null -> 0, so _height=1.
        var leaf = new TreeNode<Int, Int>(null, 1, 10, null);
        // Parent referencing the leaf: max(1,0)+1 = 2.
        var parent = new TreeNode<Int, Int>(leaf, 2, 20, null);
        var ok = leaf.get_height() == 1 && parent.get_height() == 2 && leaf.key == 1 && parent.value == 20;
        Sys.println(ok ? "PASS" : "FAIL h_leaf=" + leaf.get_height() + " h_parent=" + parent.get_height());
    }
}
