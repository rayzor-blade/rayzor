// A class instance stored through a generic context and read back from a
// generic caller keeps its own type argument at every step: the read takes
// the instance, whatever the caller's T is bound to.
class ClassThroughGenericLocalCtx {
	public static var current = new ClassThroughGenericLocalCtx();

	final values = new haxe.ds.ObjectMap<Dynamic, Dynamic>();

	public function new() {}

	public function get<T>(key:ClassThroughGenericLocalKey<T>):T {
		if (!values.exists(key))
			values.set(key, key.create());
		return cast values.get(key);
	}

	public function put<T>(key:ClassThroughGenericLocalKey<T>, v:T):Void
		values.set(key, v);
}

class ClassThroughGenericLocalKey<T> {
	public final create:Void->T;
	public var value(get, set):T;

	public function new(create:Void->T)
		this.create = create;

	function get_value():T
		return ClassThroughGenericLocalCtx.current.get(this);

	function set_value(v:T):T {
		ClassThroughGenericLocalCtx.current.put(this, v);
		return v;
	}
}

class ClassThroughGenericLocalState {
	public var scheme = 4;

	static final local = new ClassThroughGenericLocalKey<Null<ClassThroughGenericLocalState>>(() -> null);
	static var instance(get, set):Null<ClassThroughGenericLocalState>;

	static function get_instance()
		return local.value;

	static function set_instance(v:Null<ClassThroughGenericLocalState>)
		return local.value = v;

	public function new() {}

	public static function init()
		instance = new ClassThroughGenericLocalState();

	public static function tryGet():Null<ClassThroughGenericLocalState>
		return instance;
}

class ClassThroughGenericLocalNode {
	public var seen = "";

	public function new() {}

	public function style<T>(v:T):Void {
		var t = ClassThroughGenericLocalState.tryGet();
		seen += (t == null ? "null" : "" + t.scheme) + " ";
	}
}

class ClassThroughGenericLocal {
	static function main() {
		var n = new ClassThroughGenericLocalNode();
		n.style(0);
		ClassThroughGenericLocalState.init();
		n.style(1.5);
		n.style("s");
		n.style(3);
		n.style(true);
		if (n.seen != "null 4 4 4 4 ")
			throw n.seen;
		Sys.println("CONFORMANCE_OK");
	}
}
