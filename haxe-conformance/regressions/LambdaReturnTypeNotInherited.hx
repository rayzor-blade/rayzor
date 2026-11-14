// A lambda returning an object literal builds that literal, not an instance of
// the class its enclosing function returns.
private class Box {
	public var w:Float;

	public function new(w:Float, h:Float) {
		this.w = w;
	}
}

class LambdaReturnTypeNotInherited {
	static function make(flag:Bool):Box {
		var read = () -> {
			var open = flag;
			{open: open, text: open ? "t" : "", query: "q"};
		};
		var got = read();
		seen = got.open + " " + got.text + " " + got.query;
		return new Box(1, 2);
	}

	static var seen = "";

	static function main() {
		var box = make(true);
		if (seen != "true t q" || box.w != 1)
			throw seen;
		Sys.println("CONFORMANCE_OK");
	}
}
