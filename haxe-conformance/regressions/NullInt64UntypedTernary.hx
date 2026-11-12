// An untyped local holding `cond ? null : int64` is Null<Int64>: passed on to
// an Int64 parameter it is the value, not its box.
import haxe.Int64;

class Node {
	public var id:Int64;

	public function new(id) this.id = id;
}

class Tree {
	public var root:Null<Node>;

	public function new(r) root = r;
}

class NullInt64UntypedTernary {
	static function take(tree:Tree, ?root:Int64):String {
		var start = root != null ? root : tree.root == null ? null : tree.root.id;
		if (start != null)
			return visit(start);
		return "none";
	}

	static function simple(full:Bool):String {
		var v = full ? Int64.make(3, 7) : null;
		return v == null ? "none" : visit(v);
	}

	static function visit(id:Int64):String return Int64.toStr(id);

	static function main() {
		var t = new Tree(new Node(Int64.make(2, 9)));
		var got = take(t, Int64.make(1, 8)) + "|" + take(t) + "|" + take(new Tree(null)) + "|" + simple(true) + "|"
			+ simple(false);
		if (got != "4294967304|8589934601|none|12884901895|none")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
