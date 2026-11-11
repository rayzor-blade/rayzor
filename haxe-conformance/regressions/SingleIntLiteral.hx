// An Int given where a Single is expected is that Single.
typedef Opts = {?size:Single, ?wrap:Bool};

class Sized {
	public var size:Single = 16.0;

	public function new(label:String, ?o:Opts)
		size = o != null && o.size != null ? o.size : 16.0;
}

class SingleIntLiteral {
	static function size(o:Opts):String
		return o.size == null ? "null" : Std.string(Std.int(o.size * 2));

	static function main() {
		var i = 7;
		var typed:Opts = {size: 11};
		var checked = ({size: 12} : Opts);
		var plain:{size:Single} = {size: 5};
		var ctor = Std.string(Std.int(new Sized("x", {size: 28}).size));
		var got = [ctor, size({size: 28}), size({size: 28.5}), size(typed), size(checked), Std.string(Std.int(plain.size)), size({size: i}), size({})].join(",");
		if (got != "28,56,57,22,24,5,14,null")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
