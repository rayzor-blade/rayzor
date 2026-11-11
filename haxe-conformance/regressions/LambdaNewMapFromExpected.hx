// `() -> new Map()` builds the map the expected function type returns.
class Box<T> {
	public final make:Void->T;

	public function new(make:Void->T)
		this.make = make;
}

class LambdaNewMapFromExpected {
	static final shared = new Box<Map<String, Int>>(() -> new Map());

	static function take(f:Void->Map<String, String>):String {
		var m = f();
		m.set("k", "v");
		return m.get("k");
	}

	static function main() {
		var f:Void->Map<String, String> = () -> new Map();
		var m = f();
		m.set("a", "x");
		var boxed = new Box<Map<Int, Bool>>(() -> new Map()).make();
		boxed.set(1, true);
		var fromStatic = shared.make();
		fromStatic.set("n", 3);
		var got = [m.get("a"), take(() -> new Map()), "" + boxed.get(1), "" + fromStatic.get("n")].join(",");
		if (got != "x,v,true,3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
