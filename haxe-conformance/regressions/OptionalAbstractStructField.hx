// An optional structure field of an abstract type keeps a value assigned
// through the abstract's @:from, as a literal or a typed local does.
enum OptionalAbstractStructFieldRT<T> {
	Const(v:T);
	Bound(s:OptionalAbstractStructFieldSig<T>);
}

class OptionalAbstractStructFieldSig<T> {
	public var v:T;

	public function new(v:T)
		this.v = v;
}

abstract OptionalAbstractStructFieldIR<T>(OptionalAbstractStructFieldRT<T>) from OptionalAbstractStructFieldRT<T> to OptionalAbstractStructFieldRT<T> {
	@:from
	public static inline function fromConst<T>(val:T):OptionalAbstractStructFieldIR<T>
		return Const(val);

	@:from
	public static inline function fromState<T>(s:OptionalAbstractStructFieldSig<T>):OptionalAbstractStructFieldIR<T>
		return Bound(s);
}

typedef OptionalAbstractStructFieldP = {
	?reveal:OptionalAbstractStructFieldIR<Bool>
}

class OptionalAbstractStructField {
	static function kind(p:OptionalAbstractStructFieldP):String
		return switch (p.reveal : OptionalAbstractStructFieldRT<Bool>) {
			case null: "null";
			case Const(on): "const " + on;
			case Bound(s): "bound " + s.v;
		}

	static function main() {
		var s = new OptionalAbstractStructFieldSig(true);
		var props:OptionalAbstractStructFieldP = {reveal: false};
		var out = [kind(props)];
		props.reveal = s;
		out.push(kind(props));
		var empty:OptionalAbstractStructFieldP = {};
		out.push(kind(empty));
		var typed:OptionalAbstractStructFieldIR<Bool> = true;
		empty.reveal = typed;
		out.push(kind(empty));
		var got = out.join(",");
		if (got != "const false,bound true,null,const true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
