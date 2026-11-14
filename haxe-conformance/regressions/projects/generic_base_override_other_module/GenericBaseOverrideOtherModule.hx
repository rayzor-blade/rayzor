// A generic base's method overridden in another module dispatches to the
// override, from the base's own body and from a third module.
class GenericBaseOverrideOtherModule {
	static function show(b:Base<String>):String
		return b.render();

	static function main() {
		var s = new Sub("p");
		var b:Base<String> = s;
		var plain = new Base<String>("q");
		var i:Base<Int> = new SuperCaller(7);
		var got = [b.render(), show(s), b.describe(), plain.render(), plain.describe(), i.render(), i.describe(), i.keep()].join(" ");
		if (got != "sub sub describe(sub,p) base describe(base,q) over+base describe(over+base,7) keep:7")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
