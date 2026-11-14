// An object passed for a Dynamic formal stays the object: an ObjectMap keyed by
// Dynamic finds it again, as a per-context slot table does.
class OkdCtx {
	public static var current = new OkdCtx();

	public final values = new haxe.ds.ObjectMap<Dynamic, Dynamic>();

	public function new() {}

	public function get<T>(key:OkdSlot<T>):T {
		if (!values.exists(key))
			values.set(key, key.create());
		return cast values.get(key);
	}
}

class OkdSlot<T> {
	public final create:Void->T;

	public var value(get, set):T;

	public function new(create:Void->T)
		this.create = create;

	function get_value():T
		return OkdCtx.current.get(this);

	function set_value(v:T):T {
		OkdCtx.current.values.set(this, v);
		return v;
	}
}

class ObjectKeyThroughDynamicFormal {
	static final slot = new OkdSlot<Null<String>>(() -> null);

	static function main() {
		var before = slot.value == null;
		slot.value = "dark";
		var got = before + "," + slot.value;
		if (got != "true,dark")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
