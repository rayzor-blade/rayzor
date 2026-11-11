package w;

@:structInit
@:publicFields
class V {
	var id:Int;
	var name:String;
	@:optional var q:Null<Array<Int>>;
}

@:structInit
@:publicFields
class Holder {
	var v:V;
	var tag:String = "t";
}

@:structInit
class Built {
	public var total:Int;

	public function new(a:Int, b:Int)
		total = a + b;
}

class Tools {
	public static function has(v:V, x:Int):Bool {
		if (v.q != null)
			for (y in v.q)
				if (y == x)
					return true;
		return false;
	}
}

class Error {
	public var msg:String;

	public function new(msg)
		this.msg = msg;
}
