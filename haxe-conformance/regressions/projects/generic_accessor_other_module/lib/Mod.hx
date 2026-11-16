package lib;

// A generic class whose property accessors call generic methods, read from
// another module.
class Ctx {
	public static var current:Ctx = new Ctx();

	final values = new haxe.ds.ObjectMap<Dynamic, Dynamic>();

	public function new() {}

	public function get<T>(key:Local<T>):T {
		if (!values.exists(key))
			values.set(key, key.create());
		return cast values.get(key);
	}
}

class Local<T> {
	public final create:Void->T;
	public var value(get, never):T;

	public function new(create:Void->T)
		this.create = create;

	function get_value():T
		return Ctx.current.get(this);
}

class Holder {
	public var slot:Dynamic;

	public function new() {}

	public function put<T>(v:T):T {
		slot = v;
		return v;
	}

	public function take<T>():T
		return cast slot;
}

class Cell<T> {
	final h = new Holder();

	public var v(get, set):T;

	public function new() {}

	function get_v():T
		return h.take();

	function set_v(x:T):T
		return h.put(x);
}
