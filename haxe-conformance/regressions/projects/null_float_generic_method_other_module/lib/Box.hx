package lib;

class Box<S> {
	public var v = 0.0;

	public function new() {}

	public function setWithKind(kind:S, x:Float):Void
		v = x;
}
