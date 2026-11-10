interface IV {
	var ptr(default, null):Int;
	function twice():Int;
}

class Col implements IV {
	public var ptr(default, null):Int;

	public function new(p:Int)
		ptr = p;

	public function twice():Int
		return ptr * 2;
}

class Box<T> {
	var v:T;

	public function new(v:T)
		this.v = v;

	public function get():T
		return v;

	public function set(v:T):Void
		this.v = v;
}

class SV<T:IV> {
	var v:T;

	public function new(v:T)
		this.v = v;

	public function get():T
		return v;
}

class InstantiatedInterfaceArgs {
	static function check(label:String, got:Int, want:Int):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		var b = new Box<IV>(new Col(30));
		check("constructed Box<IV> field", b.get().ptr, 30);
		check("constructed Box<IV> method", b.get().twice(), 60);
		b.set(new Col(7));
		check("method on Box<IV>", b.get().ptr, 7);
		var s = new SV<IV>(new Col(40));
		check("bounded SV<IV> field", s.get().ptr, 40);
		check("bounded SV<IV> method", s.get().twice(), 80);
		var arr:Array<IV> = [new Col(5)];
		arr.push(new Col(6));
		check("array element", arr[0].ptr + arr[1].ptr, 11);
		Sys.println("CONFORMANCE_OK");
	}
}
