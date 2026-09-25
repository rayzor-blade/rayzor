// A function-local `static var` / `static final` is one variable for the
// function, initialised once and kept across calls and instances, whether
// declared at the top of the body, in a nested block or in a switch case.
class Holder {
	public function new() {}

	public function first() {
		static var last = 0;
		if (last == 0) {
			last = 1;
			return 'A';
		}
		return 'B';
	}

	public static function counter():Int {
		static var n:Int = 10;
		n++;
		return n;
	}

	public function nested(flag:Bool):Int {
		if (flag) {
			static var hits = 0;
			hits += 2;
			return hits;
		}
		return -1;
	}

	public function cased(i:Int):Int {
		switch i {
			case 0:
				static var zeros = 0;
				return ++zeros;
			case _:
				return -1;
		}
	}

	public function label():String {
		static final text = "L";
		return text;
	}
}

class FunctionLocalStatics {
	static function main() {
		var a = new Holder();
		var b = new Holder();
		var got = a.first() + b.first() + " " + Holder.counter() + "," + Holder.counter() + " "
			+ a.nested(true) + "," + b.nested(true) + "," + a.nested(false) + " "
			+ a.cased(0) + "," + b.cased(0) + " " + a.label();
		if (got == "AB 11,12 2,4,-1 1,2 L")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
