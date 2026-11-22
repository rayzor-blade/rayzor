// A generic function returning another generic call's T keeps a Float or
// Bool value exactly at a scalar T.
class Cell<T> {
	var v:T;

	public function new(v:T)
		this.v = v;

	public function get():T
		return v;
}

class Lazy<T> {
	final f:Void->T;

	public function new(f:Void->T)
		this.f = f;

	public function get():T
		return f();
}

enum Source<T> {
	Fixed(value:T);
	Held(cell:Cell<T>);
	Made(lazy:Lazy<T>);
}

class GenericReturnThroughGenericCall {
	static function read<T>(source:Source<T>):T
		return switch source {
			case Fixed(v): v;
			case Held(c): c.get();
			case Made(l): l.get();
		};

	static function direct<T>(c:Cell<T>):T
		return c.get();

	static function main() {
		var f = new Cell<Float>(0.75);
		var b = new Cell<Bool>(true);
		var i = new Cell<Int>(9);
		var got = [
			Std.string(direct(f)),
			Std.string(read(Held(f))),
			Std.string(read(Made(new Lazy<Float>(() -> 1.5 * 3)))),
			Std.string(read(Fixed(2.5))),
			Std.string(direct(b)),
			Std.string(read(Made(new Lazy<Bool>(() -> 3 > 2)))),
			Std.string(direct(i)),
			Std.string(direct(f) + 10.0)
		].join(" ");
		if (got != "0.75 0.75 4.5 2.5 true true 9 10.75")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
