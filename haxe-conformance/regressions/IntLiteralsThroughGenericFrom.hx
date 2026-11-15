abstract IR<T>(T) {
	@:from public static function fromValue<T>(v:T):IR<T>
		return cast v;

	public function get():T
		return cast this;
}

enum Rt<T> {
	Const(value:T);
}

abstract Into<T>(Rt<T>) from Rt<T> {
	@:from public static function fromConst<T>(v:T):Into<T>
		return Const(v);

	public function read():T
		return switch this {
			case Const(v): v;
		}
}

typedef S = {name:String, values:Array<Float>};

// A literal converted through `@:from f<T>(v:T)` is typed against the
// expected argument: Int literals in an Array<Float> are Floats.
class IntLiteralsThroughGenericFrom {
	static function plain(x:IR<Array<Float>>):Float
		return x.get()[0] / 2;

	static function field(x:IR<S>):Float
		return x.get().values[0] / 2;

	static function nested(x:Into<Array<S>>):Float
		return x.read()[0].values[1] / 2;

	static function main() {
		var got = [plain([85, 65]), field({name: "a", values: [85, 65]}), nested([{name: "v", values: [85, 65]}])].join(",");
		if (got != "42.5,42.5,32.5")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
