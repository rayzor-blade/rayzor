// Std.string of a class instance prints the instance whether or not it is
// typed nullable or has fields, and a null instance prints null without
// calling its toString.
class StdStringNullableInstanceState {
	public var scheme = 4;

	public function new() {}

	public function toString()
		return "State(" + scheme + ")";
}

class StdStringNullableInstanceEmpty {
	public function new() {}
}

class StdStringNullableInstance {
	static var inst:Null<StdStringNullableInstanceState> = new StdStringNullableInstanceState();

	static function get():Null<StdStringNullableInstanceState>
		return inst;

	static function main() {
		var a:Null<StdStringNullableInstanceState> = new StdStringNullableInstanceState();
		var none:Null<StdStringNullableInstanceState> = null;
		var e:StdStringNullableInstanceEmpty = new StdStringNullableInstanceEmpty();
		var ne:Null<StdStringNullableInstanceEmpty> = new StdStringNullableInstanceEmpty();
		var out = [
			Std.string(a),
			Std.string(get()),
			"" + a,
			Std.string(none),
			Std.string(e),
			Std.string(ne),
		];
		var got = out.join(",");
		if (got != "State(4),State(4),State(4),null,StdStringNullableInstanceEmpty,StdStringNullableInstanceEmpty")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
