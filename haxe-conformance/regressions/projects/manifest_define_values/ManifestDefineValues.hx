// A manifest `[build.defines]` string value is what `Context.definedValue`
// returns; `true` is the flag `1` and `false` leaves it undefined.
class ManifestDefineValues {
	static function main() {
		var got = [
			DefineValues.value("plain_flag"), DefineValues.value("my_css"), DefineValues.value("my_list"), DefineValues.value("my_num"),
			DefineValues.value("missing"), "" + DefineValues.defined("off_flag"), "" + DefineValues.defined("my_css")
		].join("|");
		#if my_css
		got += "|if";
		#end
		if (got != "1|/tmp/some/dir/Scene.css|a.css,b.css|42|<null>|false|true|if")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
