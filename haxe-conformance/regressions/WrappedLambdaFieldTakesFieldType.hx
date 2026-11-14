// A lambda written in parentheses as a structure field takes
// its untyped parameters from the field's function type, as a bare one does.
typedef Fmt = {?value:Float, ?format:Float->String}

class WrappedLambdaFieldTakesFieldType {
	static function show(p:Fmt):String
		return p.format(p.value);

	static function main() {
		var got = show({value: 0.65, format: v -> Std.string(Math.round(v * 100)) + "%"})
			+ " " + show({value: 0.65, format: (v -> Std.string(Math.round(v * 100)) + "%")});
		if (got != "65% 65%")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
