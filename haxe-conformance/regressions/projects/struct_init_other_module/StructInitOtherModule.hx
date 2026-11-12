// A structure literal for a `@:structInit` class of another module goes
// through its constructor: fields by parameter name, omitted ones defaulted.
import Shapes.Notch;
import Shapes.Edge;

class StructInitOtherModule {
	static function main() {
		var local:Notch = {a: 1, n: 2};
		var partial:Notch = {a: 3, top: Cut(1, 2)};
		var got = local.describe() + "|" + partial.describe() + "|" + show({a: 4, n: 7}) + "|"
			+ StructInitOtherModule.show({n: 9});
		if (got != "1/2/Flat/Flat|3/5/Cut(1,2)/Flat|4/7/Flat/Flat|0/9/Flat/Flat")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}

	static function show(notch:Notch):String
		return notch.describe();
}
