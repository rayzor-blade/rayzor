// A static whose module lowers after its caller's (the callee depends on the
// caller) gets its declared literal defaults for omitted arguments.
class DefaultsCalleeLoweredLater {
	public static function suffix():String {
		return "!";
	}

	static function main() {
		var got = render.Capture.shot("a", 100);
		if (got != "a 150 16777215 1 1.5 true png !")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
