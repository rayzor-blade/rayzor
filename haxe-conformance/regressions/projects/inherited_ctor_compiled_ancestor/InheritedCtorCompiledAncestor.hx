// A subclass that inherits its constructor, built before it compiles, runs the
// ancestor's constructor without changing how other modules call it.
import gen.Box;
import gen.Maker;

class InheritedCtorCompiledAncestor {
	static function main() {
		var box = new Box(7);
		var got = box.value + "," + Maker.count();
		if (got != "7,3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
