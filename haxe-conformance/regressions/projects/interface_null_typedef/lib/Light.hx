package lib;

typedef Env = {view:Int, levels:Int, intensity:Float};
typedef Shadow = {size:Int, bias:Float};

interface Lighting {
	function environment():Null<Env>;
	function shadows():Null<Shadow>;
}

class Basic implements Lighting {
	public function new() {}

	public function environment():Null<Env>
		return {view: 7, levels: 2, intensity: 1.0};

	public function shadows():Null<Shadow>
		return {size: 64, bias: 0.5};
}
