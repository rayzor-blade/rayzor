// Enum reflection through a `Dynamic` receiver. The compiler cannot inject an
// enum type id when the static type is Dynamic, so the identity has to come
// from the box — and the box tag must live in the same id space as RTTI
// registration and the ids baked at `is` sites.
enum Color {
	Red;
	Green;
	Blue(shade:Int);
}

class TestEnumReflectionDynamic {
	static var failures = 0;

	static function check(name:String, actual:String, expected:String):Void {
		if (actual != expected) {
			failures++;
			Sys.println("FAIL " + name + ": expected " + expected + " but got " + actual);
		} else {
			Sys.println("ok " + name + " = " + actual);
		}
	}

	static function main() {
		var boxed:Dynamic = Blue(3);
		var tagOnly:Dynamic = Red;

		check("isOfType(boxed)", "" + Std.isOfType(boxed, Color), "true");
		check("isOfType(tagOnly)", "" + Std.isOfType(tagOnly, Color), "true");
		check("getEnumName", Type.getEnumName(Type.getEnum(boxed)), "Color");
		check("enumConstructor(boxed)", Type.enumConstructor(boxed), "Blue");
		check("enumConstructor(tagOnly)", Type.enumConstructor(tagOnly), "Red");
		check("enumIndex(boxed)", "" + Type.enumIndex(boxed), "2");
		check("enumIndex(tagOnly)", "" + Type.enumIndex(tagOnly), "0");

		// A statically-typed receiver must keep working — it takes the
		// compiler-injected id path rather than the box.
		var stat:Color = Blue(7);
		check("static enumConstructor", Type.enumConstructor(stat), "Blue");
		check("static enumIndex", "" + Type.enumIndex(stat), "2");

		// A non-enum Dynamic must not be mistaken for one.
		var notEnum:Dynamic = "hello";
		check("isOfType(String, Color)", "" + Std.isOfType(notEnum, Color), "false");

		if (failures == 0) {
			Sys.println("ALL PASS");
		} else {
			Sys.println("FAILURES: " + failures);
		}
	}
}
