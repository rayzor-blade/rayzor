import lib.Value;
import lib.Writer;

class MethodRefGenericOtherModule {
	static function main() {
		var w = new Writer();
		var typed:(Int, Value<Int>) -> Void = w.apply;
		typed(1, 5);
		var floats:(Int, Value<Float>) -> Void = w.apply;
		floats(2, 2.5);
		var later:(Int, Value<String>) -> Void = w.apply;
		later(3, Bound(() -> "x"));
		var putFloat:(Int, Float) -> Void = w.put;
		putFloat(4, 1.5);
		var putString:(Int, String) -> Void = w.put;
		putString(5, "s");
		var got = w.log.join(" ");
		if (got != "1=5 2=2.5 3=x 4:1.5 5:s")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
