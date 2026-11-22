package lib;

interface Read<T> {
	function get():T;
}

class Kept<T> implements Read<T> {
	var last:T;
	final compute:Void->T;

	public function new(compute:Void->T) {
		this.compute = compute;
		last = compute();
	}

	public function get():T
		return last;
}

class Make {
	public static function of<T>(compute:Void->T):Read<T>
		return new Kept<T>(compute);
}
