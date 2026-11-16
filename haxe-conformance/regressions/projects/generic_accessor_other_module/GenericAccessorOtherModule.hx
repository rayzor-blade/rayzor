import lib.Mod;

class GenericAccessorOtherModule {
	static final names = new Local<String>(() -> "hi");
	static final lists = new Local<Array<String>>(() -> ["a", "b"]);

	static function main() {
		var out = [names.value, Std.string(lists.value.length)];
		var s = new Cell<String>();
		s.v = "abc";
		var f = new Cell<Float>();
		f.v = 2.5;
		out.push(s.v + " " + s.v.length + " " + (f.v * 2));
		var got = out.join(",");
		if (got != "hi,2,abc 3 5")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
