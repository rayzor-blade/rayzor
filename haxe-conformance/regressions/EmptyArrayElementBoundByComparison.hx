// Comparing an untyped empty array's element with a String binds the element
// to String before the first push, so the comparison is by value.
class EmptyArrayElementBoundByComparison {
	static function main() {
		var f = [];
		var out = [];
		for (t in "..,..,x".split(",")) {
			out.push(f.length > 0 && f[f.length - 1] != "..");
			f.push(t);
		}
		var got = out.join(",") + " " + haxe.io.Path.normalize("../../x/y") + " "
			+ haxe.io.Path.join(["../../a/b", "c.bin"]);
		if (got != "false,false,false ../../x/y ../../a/b/c.bin")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
