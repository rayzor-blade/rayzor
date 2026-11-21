package lib;

import lib.Form;

class Element {
	public var tree(default, null):Tree;
	public var given(default, null):String;

	public function new(?tree:Tree) {
		given = tree == null ? "null" : tree.name;
		this.tree = tree != null ? tree : Tree.current;
		Form.seen++;
	}
}
