class MacroTypedMacroEnum {
	static function main() {
		var bare = z.Mac.bare(7);
		var qualified = z.Mac.qualified(8);
		var names = z.Mac.names();
		var anon = z.Mac.anon();
		var got = bare.v + "," + qualified.v + "," + names + "," + anon;
		var want = "7,8,haxe.macro.ComplexType:TPath haxe.macro.ComplexType:TPath," + [for (i in 0...3) "var x:{ var a : Int; var b : Int; }"].join("|");
		if (got != want)
			throw got + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
