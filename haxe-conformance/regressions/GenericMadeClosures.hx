// A closure built inside a specialized generic body is called natively by a
// caller that knows the type argument, and through an erased view by generic
// code and the runtime's array helpers.
abstract GenericMadeClosuresTwice<T>(T->T) {
	inline function new(f) this = f;

	@:from public static function convert<T>(value:T->T):GenericMadeClosuresTwice<T> {
		var once = value;
		return new GenericMadeClosuresTwice((x:T) -> once(once(x)));
	}

	public function apply(v:T):T
		return this(v);
}

class GenericMadeClosures {
	static function twice<T>(f:T->T):T->T
		return (x:T) -> f(f(x));

	static function apply<T>(f:T->T, v:T):T
		return f(v);

	static function main() {
		var out = [];
		var q:Float->Float = twice((v:Float) -> v + 0.25);
		out.push(Std.string(q(1.0)));
		out.push(Std.string(apply(twice((v:Float) -> v * 3), 0.5)));
		out.push(Std.string([1.0, 2.0].map(twice((v:Float) -> v * 10))));
		var i:Int->Int = twice((v:Int) -> v * 7);
		out.push(Std.string(i(2)));
		out.push(apply(twice((s:String) -> s + "!"), "hi"));
		out.push(Std.string(apply(twice((b:Bool) -> !b), true)));
		var c:GenericMadeClosuresTwice<Float> = (v:Float) -> v + 0.25;
		out.push(Std.string(c.apply(1.0)));
		var got = out.join(",");
		if (got != "1.5,4.5,[100,200],98,hi!!,true,1.5")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
