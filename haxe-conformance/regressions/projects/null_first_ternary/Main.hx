class Main {
	static function main() {
		var b = Std.random(1);
		var kept = b == 7 ? null : "abc";
		if (e.Runtime.haxe_string_len(kept) != 3) throw "variable";
		if (e.Runtime.haxe_string_len(b == 7 ? null : "abcd") != 4) throw "inline";
		var none = b == 0 ? null : "abc";
		if (none != null) throw "null branch taken";
		Sys.println("CONFORMANCE_OK");
	}
}
