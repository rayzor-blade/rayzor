@:build(lib.Mac.build())
class Holder {}

class MacroFieldImports {
	static function main() {
		var want = "<hello><x>3";
		if (Holder.TEXT != want)
			throw Holder.TEXT + " != " + want;
		if (lib.User.run() != want)
			throw lib.User.run() + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
