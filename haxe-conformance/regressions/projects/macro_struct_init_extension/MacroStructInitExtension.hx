@:build(w.Mac.build())
class Out {}

class MacroStructInitExtension {
	static function main() {
		var want = "true,false,false,true,true,true,t,w.V,w.V,7,false,true,k!!,other";
		if (Out.TEXT != want)
			throw Out.TEXT + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
