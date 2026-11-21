// A `Null<A>` local, field or argument takes A's `@:from` once, as an `A` one
// does; `null` stays null.
enum RT<T> {
	Const(value:T);
	Bound(sig:Sig<T>);
}

class Sig<T> {
	public var value:T;

	public function new(value:T)
		this.value = value;
}

abstract IR<T>(RT<T>) from RT<T> to RT<T> {
	@:from
	public static inline function fromConst<T>(val:T):IR<T>
		return Const(val);

	@:from
	public static inline function fromState<T>(s:Sig<T>):IR<T>
		return Bound(s);
}

class Holder {
	public var plain:IR<Bool>;
	public var opt:Null<IR<Bool>>;

	public function new() {}
}

class NullableAbstractFromAssignments {
	static function show(v:Null<IR<Bool>>):String
		return switch (v : RT<Bool>) {
			case null: "null";
			case Const(x): "const " + x;
			case Bound(s): "bound " + s.value;
		}

	static function main() {
		var s = new Sig(true);
		var c = new Holder();
		c.plain = s;
		var local:Null<IR<Bool>> = s;
		var lit:Null<IR<Bool>> = false;
		var none:Null<IR<Bool>> = null;
		var out = [show(c.plain), show(local), show(lit), show(none)];
		c.opt = true;
		out.push(show(c.opt));
		c.opt = s;
		out.push(show(c.opt));
		c.opt = null;
		out.push(show(c.opt));
		out.push(show(true));
		out.push(show(s));
		var got = out.join(",");
		if (got != "bound true,bound true,const false,null,const true,bound true,null,const true,bound true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
