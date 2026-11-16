// A generic class's property accessor specializes with the receiver's type
// arguments, so a getter or setter that calls a generic method moves T by
// its own type.
class GenericPropertyAccessorsCtx {
	public static var current = new GenericPropertyAccessorsCtx();

	final values = new haxe.ds.ObjectMap<Dynamic, Dynamic>();

	public function new() {}

	public function get<T>(key:GenericPropertyAccessorsLocal<T>):T {
		if (!values.exists(key))
			values.set(key, key.create());
		return cast values.get(key);
	}
}

class GenericPropertyAccessorsLocal<T> {
	public final create:Void->T;
	public var value(get, never):T;

	public function new(create:Void->T)
		this.create = create;

	function get_value():T
		return GenericPropertyAccessorsCtx.current.get(this);
}

class GenericPropertyAccessorsHolder {
	public var slot:Dynamic;

	public function new() {}

	public function put<T>(v:T):T {
		slot = v;
		return v;
	}

	public function take<T>():T
		return cast slot;
}

class GenericPropertyAccessorsCell<T> {
	final h = new GenericPropertyAccessorsHolder();

	public var v(get, set):T;

	public function new() {}

	function get_v():T
		return h.take();

	function set_v(x:T):T
		return h.put(x);
}

class GenericPropertyAccessors {
	static final names = new GenericPropertyAccessorsLocal<String>(() -> "hi");
	static final lists = new GenericPropertyAccessorsLocal<Array<String>>(() -> ["a", "b"]);

	static function main() {
		var out = [names.value, Std.string(lists.value.length)];
		var s = new GenericPropertyAccessorsCell<String>();
		s.v = "abc";
		var f = new GenericPropertyAccessorsCell<Float>();
		f.v = 2.5;
		var a = new GenericPropertyAccessorsCell<Array<Int>>();
		a.v = [1, 2, 3];
		out.push(s.v + " " + s.v.length + " " + (f.v * 2) + " " + a.v.length);
		var r = (s.v = "xy");
		out.push(r + " " + s.v);
		var got = out.join(",");
		if (got != "hi,2,abc 3 5 3,xy xy")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
