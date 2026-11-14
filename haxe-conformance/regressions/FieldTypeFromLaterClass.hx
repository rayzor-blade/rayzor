// A method reads a field of a class declared after it: the singleton's type
// and its typed `new` field are known before the later class lowers.
class LaterSlot {
	public function new() {}

	public function put(v:String)
		LaterCtx.current.values.set(this, v);

	public function count():Int
		return LaterCtx.current.names.length;
}

class LaterCtx {
	public static var current = new LaterCtx();

	public final values = new haxe.ds.ObjectMap<Dynamic, Dynamic>();
	public var names = ["a", "b"];

	public function new() {}
}

class FieldTypeFromLaterClass {
	static function main() {
		var s = new LaterSlot();
		s.put("x");
		var got = LaterCtx.current.values.get(s) + " " + s.count();
		if (got != "x 2")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
