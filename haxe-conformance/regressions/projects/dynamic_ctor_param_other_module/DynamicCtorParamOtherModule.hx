// A constructor from another module boxes a concrete argument for its
// Dynamic parameter, as one in the same module does.
class DynamicCtorParamOtherModule {
	static function main() {
		var s = "ab";
		var held = [new lib.Holder(s, "s"), new lib.Holder(3, "i"), new lib.Holder(1.5, "f"), new lib.Holder(true, "b")];
		var got = [for (h in held) h.label + "=" + h.value].join(" ") + " " + Std.string(new lib.Holder(s, "").value);
		if (got != "s=ab i=3 f=1.5 b=true ab")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
