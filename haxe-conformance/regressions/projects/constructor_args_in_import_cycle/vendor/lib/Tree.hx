package lib;

class Tree {
	public static var current:Tree = new Tree("root");

	public final name:String;

	public function new(name:String)
		this.name = name;
}
