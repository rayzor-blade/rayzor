// A module with more ids than one import block keeps its own; the next
// module's ids start past it.
import lib.After;
import lib.Many;

class ImportIdBlocks {
	static function main() {
		var got = Many.f0() + "," + Many.f10049() + "," + After.twice(21);
		if (got != "0,10049,42")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
