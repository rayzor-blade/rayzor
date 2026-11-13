// A macro's output calling another macro by its qualified path inside a
// comprehension expands that call too.
class MacroQualifiedInComprehension {
	static function main() {
		var got = [Std.string(pk.M.direct()), Std.string(pk.M.outer(0)), Std.string(pk.M.withPos())].join(" ");
		if (got != "40 [0,10,20] 50")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
