package a;

class Prop<T> {
	public var id:Int;

	public function new(id:Int)
		this.id = id;
}

class Node {
	public var log:Array<String> = [];

	public function new() {}

	public function set<T>(prop:Prop<T>, value:T):Void
		log.push(prop.id + "=" + value);
}
