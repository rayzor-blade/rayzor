// A Dynamic string handed to a StringTools function mapped to the runtime is
// opened from its box first, as a String local is.
class DynamicArgumentToStringTools {
	static function main() {
		var d:Dynamic = haxe.Json.parse('{"uri":"data:abc","n":"x.png"}');
		var got = [
			Std.string(StringTools.startsWith(d.uri, "data:")),
			Std.string(StringTools.endsWith(d.n, ".png")),
			StringTools.replace(d.n, ".png", ".dds"),
			StringTools.urlDecode(d.n),
		].join(" ");
		if (got != "true true x.dds x.png")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
