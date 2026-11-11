// `/` in a macro is Float division, Int operands included.
class MacroFloatDivision {
	static function main() {
		var got = DivMac.ratios();
		if (got != "0.2,0.2,3.5,1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
