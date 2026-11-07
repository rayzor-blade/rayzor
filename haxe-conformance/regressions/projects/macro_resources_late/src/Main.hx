class Main {
	static function main() {
		if (r.One.get() != "first") throw "first resource";
		if (r.Two.get() != "second") throw "resource added after the table was compiled";
		var names = haxe.Resource.listNames();
		names.sort((a, b) -> a < b ? -1 : a > b ? 1 : 0);
		if (names.join(",") != "one,two") throw "resource names";
		Sys.println("CONFORMANCE_OK");
	}
}
