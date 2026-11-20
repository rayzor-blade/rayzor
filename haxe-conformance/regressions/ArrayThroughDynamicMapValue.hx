// An array stored as a Dynamic map value, by `set` or by `m[k] = v`, is read
// back as an array: its Dynamic carries the array's box.
class ArrayThroughDynamicMapValue {
	static function id<T>(v:Dynamic):T
		return cast v;

	function new() {}

	static function main() {
		var out = [];
		var m = new haxe.ds.ObjectMap<Dynamic, Dynamic>();
		var k = new ArrayThroughDynamicMapValue(), k2 = new ArrayThroughDynamicMapValue();
		var arr = [1, 2, 3];
		m.set(k, arr);
		m.set(k2, ["x"]);
		var c:Dynamic = m.get(k);
		var g:Dynamic = m.get(k2);
		out.push(c.length);
		out.push(g.length);
		out.push(id(g).length);
		var s = new Map<String, Dynamic>();
		s["x"] = [4, 5];
		var x:Dynamic = s["x"];
		out.push(x.length);
		var got = out.join(",") + " " + Std.string(c) + " " + Std.string(x);
		if (got != "3,1,1,2 [1,2,3] [4,5]")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
