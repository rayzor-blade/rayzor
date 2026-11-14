// A Dynamic converts to the concrete type it meets: an array index, a scalar
// extern parameter, a typed local initialised from a field, a typed return.
class DynamicToConcreteConversions {
	static function asArray(v:Dynamic):Array<Dynamic>
		return v;

	static function asInt(v:Dynamic):Int
		return v;

	static function asString(v:Dynamic):String
		return v;

	static var traced = 0;

	static function count():Int
		return ++traced;

	static function main() {
		var o:Dynamic = {n: [1, 2, 3], i: 1, f: 1.5, s: "abcd"};
		var arr = ["a", "b", "c"];
		var d:Dynamic = o.i;
		var nodes:Array<Dynamic> = o.n;
		var text:String = o.s;
		var json:Dynamic = haxe.Json.parse('{"nodes":[1,2,3],"scene":1}');
		var parsed:Array<Dynamic> = json.nodes;
		var walked = 0;
		for (_ in parsed)
			walked++;
		trace(count(), count());
		var got = [
			arr[d], arr[o.i], arr[json.scene != null ? json.scene : 0],
			"" + Std.int(o.f), "" + Math.floor(o.f), "" + Std.int(d),
			"" + nodes.length, "" + text.length, "" + parsed.length, "" + walked,
			"" + asArray(o.n).length, "" + (asInt(o.i) + 1), "" + asString(o.s).length,
			"" + traced
		].join(",");
		if (got != "b,b,b,1,1,1,3,4,3,3,3,2,4,2")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
