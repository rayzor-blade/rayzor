// A static call lowered before its callee's module still converts its arguments to the declared types.
class CallBeforeCalleeLowered {
	static function main() {
		var got = new comp.Badge("success").result;
		if (got != "badge:size=md;variant=success")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
