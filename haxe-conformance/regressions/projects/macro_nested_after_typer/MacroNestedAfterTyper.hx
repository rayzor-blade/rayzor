// A macro that consults the typer returns code calling imported macros of
// its own class; those calls expand, nested ones too.
import NestedMac.outerTyped;
import NestedMac.wrap;

class MacroNestedAfterTyper {
	static function main() {
		var f = () -> outerTyped("wrap('tl')");
		var got = outerTyped("wrap('t')") + "," + f() + "," + outerTyped("wrap(outerTyped(\"wrap('n')\"))");
		if (got != "<[t]>,<[tl]>,<[<[n]>]>")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
