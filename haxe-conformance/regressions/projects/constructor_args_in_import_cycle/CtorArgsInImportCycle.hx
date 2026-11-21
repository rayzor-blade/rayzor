// A constructor call compiled before its class's module, inside an import
// cycle, passes null for each optional argument it omits.
import lib.Form;

class CtorArgsInImportCycle {
	static function main() {
		var got = Check.run();
		if (got != "null/null null/null root/3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
