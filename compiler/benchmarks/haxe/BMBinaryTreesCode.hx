// Binary trees -- the allocation benchmark of the heavy set, taken from
// HashLink's own suite (other/benchs/BinaryTrees.hx in HaxeFoundation/hashlink,
// itself the Benchmarks Game kernel).
//
// The set had a hole here: mandelbrot and nbody are FP, fib and the call
// kernels are call latency, and since SROA started scalarizing mandelbrot's
// per-iteration objects NOTHING in the suite allocates on the hot path. This
// does almost nothing else: bottomUpTree builds millions of short-lived
// perfect trees the collector must reclaim while one long-lived tree stays
// rooted across the whole run, which is precisely the young-garbage-plus-
// tenured-survivor shape a Heaps frame produces and the shape a bump
// allocator + block collector is worst at faking its way through.
//
// itemCheck's recursive walk keeps the trees honest -- a tree the GC
// corrupted changes the checksum rather than merely crashing.
//
// n = 16 rather than upstream's 14, so the compiled tiers spend seconds here
// the way they do on the other heavy rows (upstream's n=14 checksum 26208
// was reproduced first to validate the port, then n was raised; the n=16
// checksum below is cross-validated between ash's interpreter and stock
// HashLink 1.15 on x86_64).
class TreeNode {
    var left:TreeNode;
    var right:TreeNode;
    var item:Int;

    public function new(left, right, item) {
        this.left = left;
        this.right = right;
        this.item = item;
    }

    public function itemCheck():Int {
        if (left == null) return item;
        return item + left.itemCheck() - right.itemCheck();
    }
}

class BMBinaryTreesCode {
    static function bottomUpTree(item:Int, depth:Int):TreeNode {
        if (depth > 0)
            return new TreeNode(
                bottomUpTree(2 * item - 1, depth - 1),
                bottomUpTree(2 * item, depth - 1),
                item
            );
        return new TreeNode(null, null, item);
    }

    static function main() {
        var minDepth = 4;
        var n = 16;
        var maxDepth = Std.int(Math.max(minDepth + 2, n));
        var stretchDepth = maxDepth + 1;
        var check = bottomUpTree(0, stretchDepth).itemCheck();
        var result = check;

        var longLivedTree = bottomUpTree(0, maxDepth);
        var depth = minDepth;
        while (depth <= maxDepth) {
            var iterations = 1 << (maxDepth - depth + minDepth);
            check = 0;
            for (i in 0...iterations) {
                check += bottomUpTree(i, depth).itemCheck();
                check += bottomUpTree(-i, depth).itemCheck();
            }
            result ^= check;
            depth += 2;
        }

        result ^= longLivedTree.itemCheck();
        Sys.println("Checksum: " + result);
    }
}
