// A generic method's Null<T> result at T = Bool or Float keeps its null.
class Maybe<T> {
	final make:Float->Null<T>;

	public function new(make:Float->Null<T>)
		this.make = make;

	public function at(t:Float):Null<T>
		return make(t);
}

class GenericNullableScalarResult {
	static function main() {
		var b = new Maybe<Bool>(t -> t > 0.5 ? true : null);
		var f = new Maybe<Float>(t -> t > 0.5 ? t * 2 : null);
		var got = [
			Std.string(b.at(0.25)),
			Std.string(b.at(0.75)),
			Std.string(f.at(0.25)),
			Std.string(f.at(0.75)),
			Std.string(b.at(0.25) == null),
			Std.string(f.at(0.75) == 1.5)
		].join(" ");
		if (got != "null true null 1.5 true true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
