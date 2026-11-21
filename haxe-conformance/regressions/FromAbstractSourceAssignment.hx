// An assignment converts a value of a generic abstract by the target abstract's
// @:from taking that abstract, as an initialiser, an argument or a return does.
class Box<T> {
	public var v:T;

	public function new(v:T)
		this.v = v;
}

// An abstract source type: the conversion below takes Sig<T>, not Box<T>.
abstract Sig<T>(Box<T>) from Box<T> to Box<T> {
	public static function make<T>(v:T):Sig<T>
		return new Box(v);

	public function get():T
		return this.v;
}

enum RT<T> {
	Const(value:T);
	Bound(state:Sig<T>);
}

abstract IR<T>(RT<T>) from RT<T> to RT<T> {
	@:from
	public static inline function fromConst<T>(val:T):IR<T>
		return Const(val);

	@:from
	public static inline function fromState<T>(state:Sig<T>):IR<T>
		return Bound(state);
}

typedef FromAbstractSourceAssignmentP = {reveal:IR<Bool>}

typedef FromAbstractSourceAssignmentQ = {?reveal:IR<Bool>}

class Holder {
	public var field:IR<Bool> = Const(false);

	public function new() {}
}

class FromAbstractSourceAssignment {
	static function kind(r:IR<Bool>):String
		return switch (r : RT<Bool>) {
			case null: "null";
			case Const(on): "const " + on;
			case Bound(s): "bound " + s.get();
		}

	static function take(r:IR<Bool>):String
		return kind(r);

	static function give(s:Sig<Bool>):IR<Bool>
		return s;

	static function main() {
		var s = Sig.make(true);
		var out = [];
		var local:IR<Bool> = s;
		out.push(kind(local));
		var reassigned:IR<Bool> = Const(false);
		reassigned = s;
		out.push(kind(reassigned));
		var p:FromAbstractSourceAssignmentP = {reveal: false};
		p.reveal = s;
		out.push(kind(p.reveal));
		var q:FromAbstractSourceAssignmentQ = {};
		q.reveal = s;
		out.push(kind(q.reveal));
		var h = new Holder();
		h.field = s;
		out.push(kind(h.field));
		out.push(take(s));
		out.push(kind(give(s)));
		var got = out.join(",");
		if (got != "bound true,bound true,bound true,bound true,bound true,bound true,bound true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
