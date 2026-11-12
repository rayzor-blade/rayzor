// `Input` imported by this file is lib.Input, not the standard library's
// haxe.io.Input, bare or qualified.
import lib.Input;

class ImportedNameOverStdType {
	static function main() {
		var got = new Input({value: "a"}).render() + " " + new Input({value: "b"}, "x").tag + " "
			+ new lib.Input({value: "q"}).render();
		if (got != "input[a] x input[q]")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
