package p;
class Item {
	public var on = false;
	public var tree:Tree;
	public function new(tree:Tree) this.tree = tree;
	function setState(v:Bool) on = v;
	public inline function setOn(v:Bool):Void setState(v);
	public inline function isOn():Bool return on;
	public function focus():Void Focus.set(this);
}
