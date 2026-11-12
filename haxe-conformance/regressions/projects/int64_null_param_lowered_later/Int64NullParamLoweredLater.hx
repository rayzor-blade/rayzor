// An Int64 passed to a Null<Int64> parameter of a static whose module lowers
// after the caller's arrives as the value, and null as null.
import haxe.Int64;

class Int64NullParamLoweredLater {
	public static function tag():String {
		return "!";
	}

	static function main() {
		var id = Int64.make(1, 0x38);
		var got = dbg.Trace.begin(id, "a") + "," + dbg.Trace.describe(id) + "," + dbg.Trace.begin(null, "b");
		if (got != "a:4294967352!,4294967352,b:?!")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
