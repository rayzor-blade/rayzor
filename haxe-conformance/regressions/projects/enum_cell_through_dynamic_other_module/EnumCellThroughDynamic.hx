import lib.Shapes;

// An enum value held as Dynamic passes to a parameter of its enum type
// from another module, whatever its constructor's index.
class EnumCellThroughDynamic {
	static function main() {
		var all:Array<Dynamic> = [A5(new Box(1)), B5(new Box(2)), C5(new Box(3)), D5(new Box(4)), E5(new Box(5))];
		var shown = [];
		for (d in all)
			shown.push(Shapes.show(d));
		var got = shown.join(" ");
		var one:Dynamic = C5(new Box(6));
		got += " " + Shapes.show(one) + " " + Std.string(Type.enumIndex(one));
		if (got != "A1 B2 C3 D4 E5 C6 2")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
