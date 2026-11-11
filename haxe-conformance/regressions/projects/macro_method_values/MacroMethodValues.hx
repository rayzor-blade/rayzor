@:build(MethodValuesMac.build())
class MethodValuesHolder {}

class MacroMethodValues {
	static function main() {
		var want = "1111,false,false";
		if (MethodValuesHolder.TEXT != want)
			throw MethodValuesHolder.TEXT + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
