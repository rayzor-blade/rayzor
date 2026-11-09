package shapes;

interface IV {
	var ptr(default, null):Int;
}

class Col implements IV {
	public var ptr(default, null):Int;
	public final rgb:Int;

	public function new(rgb:Int) {
		this.rgb = rgb;
		this.ptr = rgb + 1;
	}
}

class SBField<T:IV> {
	public var current:T;

	public function new(v:T)
		current = v;
}

interface IS<T> {
	function get():T;
}

class SV<T:IV> implements IS<T> {
	var current:T;

	public function new(v:T)
		current = v;

	public function get():T
		return current;

	public function ptrOf():Int
		return current.ptr;
}

interface IPlain {
	function id():Int;
}

class PlainImpl implements IPlain {
	public function new() {}

	public function id():Int
		return 77;
}

enum RP {
	BP(s:IPlain);
}

interface ISg<T> {
	var ptr(default, null):Int;
	function get():T;
}

class SgI implements ISg<Int> {
	public var ptr(default, null):Int;

	var v:Int;

	public function new(v:Int) {
		this.v = v;
		ptr = 77;
	}

	public function get():Int
		return v;
}

@:forward(ptr, get)
abstract Sg<T>(ISg<T>) from ISg<T> to ISg<T> {}

enum RS {
	BS(s:Sg<Int>);
}
