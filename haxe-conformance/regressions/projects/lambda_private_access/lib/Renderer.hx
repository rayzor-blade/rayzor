package lib;

class Target {
	public var view:Int;
	public var texture:Int;
	public function new(v) {
		view = v;
		texture = v * 2;
	}
}

class Renderer {
	var mask:Null<Target>;
	public function new() {}

	public function maskTarget(w:Int):Int {
		var at = mask;
		if (at == null || at.texture != w * 2)
			mask = at = new Target(w);
		return at.view;
	}

	function beginPass(encoder:Int, view:Int):Int
		return encoder + view;
}
