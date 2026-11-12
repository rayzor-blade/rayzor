// A packaged class's `static final m = new Map<K, V>()` builds the concrete
// map its key selects, though nothing imports that map.
import lib.Doc;
import lib.Reg;

class StaticNewMapPackaged {
	static function main() {
		var d = Doc.make("x");
		var r = Reg.make(2);
		Reg.make(3);
		var got = d.id + " " + (Doc.make("x") == d) + "|" + r.id + " " + (Reg.make(2) == r) + " " + Reg.describe(r);
		if (got != "7 true|2 true r2 2 3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
