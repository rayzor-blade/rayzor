class MacroAbstractThis {
	static function main() {
		var got = AbstractThisMac.run();
		var want = "123/456/truefalse/falsefalsefalsefalse,0,1,true,2,1,true";
		if (got != want)
			throw got + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
