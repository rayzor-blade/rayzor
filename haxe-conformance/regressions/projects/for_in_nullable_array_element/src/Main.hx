// A loop over a Null<Array<Item>> binds Item, so a static extension applies to it.
class Main {
	static function main() {
		var t = new p.Tree();
		var a = new p.Item(t), b = new p.Item(t);
		a.focus();
		b.focus();
		var got = a.on + " " + b.on;
		if (got != "false true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
