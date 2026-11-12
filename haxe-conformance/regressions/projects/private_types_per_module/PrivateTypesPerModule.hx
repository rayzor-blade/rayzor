// Same-named private types of two modules are each module's own: statics,
// methods, instances and typedef shapes stay apart.
class PrivateTypesPerModule {
	static function main() {
		var got = PartA.describe() + "|" + PartB.describe();
		if (got != "A 3 true 8 20 a3|B 1 true twice4 bb 1.5xy")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
