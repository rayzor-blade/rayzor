// An enum abstract's values are bare names only where its module is imported.
class EnumAbstractValueHiddenWithoutImport {
	static function main() {
		new Option("x").render();
		var got = Select.seen.join(",") + " " + (Base.key : Int);
		if (got != "x:box 13")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
