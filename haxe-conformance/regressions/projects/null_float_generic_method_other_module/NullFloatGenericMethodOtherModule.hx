import lib.Box;

// A `Null<Float>` argument for a Float formal of a generic class's method from
// another module arrives as its value.
class NullFloatGenericMethodOtherModule {
	static function main() {
		final d:Null<Float> = 0.3;
		final unset:Null<Float> = null;
		final b = new Box<Int>();
		b.setWithKind(1, d);
		final first = b.v;
		b.setWithKind(2, unset == null ? 0.5 : unset);
		final got = first + " " + b.v;
		if (got != "0.3 0.5")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
