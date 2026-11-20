// A static generic put/take through an ObjectMap<Dynamic, Dynamic> keys a
// class instance by itself however often it is boxed, and a value read back
// at a reference type is the reference, not its box.
class GenericStaticObjectMapStoreStore {
	public static var map = new haxe.ds.ObjectMap<Dynamic, Dynamic>();

	public static function put<T>(k:Dynamic, v:T):T {
		map.set(k, v);
		return v;
	}

	public static function take<T>(k:Dynamic):T
		return cast map.get(k);
}

class GenericStaticObjectMapStoreCell<T> {
	public var v(get, set):T;

	public function new() {}

	function get_v():T
		return GenericStaticObjectMapStoreStore.take(this);

	function set_v(x:T):T
		return GenericStaticObjectMapStoreStore.put(this, x);
}

class GenericStaticObjectMapStore {
	static function put<T>(m:haxe.ds.ObjectMap<Dynamic, Dynamic>, k:Dynamic, v:T):Void
		m.set(k, v);

	static function take<T>(m:haxe.ds.ObjectMap<Dynamic, Dynamic>, k:Dynamic):T
		return cast m.get(k);

	function new() {}

	static function main() {
		var s = new GenericStaticObjectMapStoreCell<String>();
		s.v = "abc";
		var f = new GenericStaticObjectMapStoreCell<Float>();
		f.v = 2.5;
		var a = new GenericStaticObjectMapStoreCell<Array<Int>>();
		a.v = [1, 2, 3];
		var out = [s.v + " " + s.v.length + " " + (f.v * 2) + " " + a.v.length];
		var r = (s.v = "xy");
		out.push(r + " " + s.v);

		var m = new haxe.ds.ObjectMap<Dynamic, Dynamic>();
		var key = new GenericStaticObjectMapStore();
		put(m, key, new Array<String>());
		var list:Array<String> = take(m, key);
		out.push(Std.string(list.length));
		list.push("x");
		var again:Array<String> = take(m, key);
		out.push(Std.string(again.length));

		var got = out.join(",");
		if (got != "abc 3 5 3,xy xy,0,1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
