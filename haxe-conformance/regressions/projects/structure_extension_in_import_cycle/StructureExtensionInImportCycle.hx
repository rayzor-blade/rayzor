// A typedef extending a type of a module in the same import cycle is typed
// after that module, so a literal of it converts each field by its base type.
import lib.ui.Form;

class StructureExtensionInImportCycle {
	static function main() {
		var tag = Check.revealTag();
		if (tag != "const true")
			throw tag;
		Sys.println("CONFORMANCE_OK");
	}
}
