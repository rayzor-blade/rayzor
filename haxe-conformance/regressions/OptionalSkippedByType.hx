class Tree {
	public function new() {}
}

class Base {
	public var tree:Tree;

	public function new(?tree:Tree) {
		this.tree = tree;
	}
}

class Box extends Base {
	public var kids:Array<Base>;

	public function new(?attr:{height:Int}, ?children:Array<Base>, ?tree:Tree) {
		super(tree);
		kids = children;
	}
}

class OptionalSkippedByType {
	static function call(?a:Int, ?list:Array<Int>, ?tree:Tree):String
		return "list " + (list != null) + " tree " + (tree != null);

	static function main() {
		var t = new Tree();
		var b = new Box({height: 50}, t);
		if (b.tree == null || b.kids != null) throw "constructor binds the tree to children";
		if (call(1, t) != "list false tree true") throw "call(1, t)";
		if (call(t) != "list false tree true") throw "call(t)";
		if (call(1, [2], t) != "list true tree true") throw "all bound";
		if (call() != "list false tree false") throw "none bound";
		Sys.println("CONFORMANCE_OK");
	}
}
