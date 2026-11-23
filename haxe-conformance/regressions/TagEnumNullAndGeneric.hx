// A tag-only enum's first constructor is a value, not null, wherever the enum
// is nullable or bound to a type parameter, and a generic body prints it by
// name.
enum Phase {
	Closed;
	Waiting;
	Open;
}

enum Signal {
	Rest;
	Leave;
	Timeout;
}

class Cell<T> {
	var value:T;

	public function new(v:T)
		value = v;

	public function get():T
		return value;

	public function set(v:T):Void
		value = v;
}

class Machine<S:EnumValue, E> {
	public final state:Cell<S>;
	final transition:(S, E) -> Null<S>;

	public final log:Array<String> = [];

	public function new(initial:S, transition:(S, E) -> Null<S>) {
		state = new Cell<S>(initial);
		this.transition = transition;
	}

	public function send(event:E):Void {
		final from = state.get();
		final to = transition(from, event);
		log.push(Std.string(event) + ":" + Std.string(from) + ">" + Std.string(to));
		if (to != null && !Type.enumEq(to, from))
			state.set(to);
	}
}

class TagEnumNullAndGeneric {
	static function isNull<T>(v:Null<T>):Bool
		return v == null;

	static function str<T>(v:T):String
		return Std.string(v);

	static function pick<T>(a:T, first:Bool):Null<T>
		return first ? a : null;

	static function index<T:EnumValue>(v:T):Int
		return Type.enumIndex(v);

	static function same<T>(a:T, b:T):Bool
		return a == b;

	static function main() {
		final out = [];
		out.push(isNull(Closed) + "," + isNull(Waiting) + "," + isNull((null : Phase)));
		out.push(str(Closed) + "," + str(Waiting) + "," + str(Open));
		final p = pick(Closed, true);
		final q = pick(Closed, false);
		out.push((p != null) + "," + Std.string(p) + "," + (q == null));
		final n:Null<Phase> = Closed;
		final s:Phase = n;
		out.push((n == null) + "," + Std.string(n) + "," + Std.string(s));
		final d:Dynamic = Closed;
		final e:Dynamic = Closed;
		out.push(Std.string(d) + "," + (d == null) + "," + (d == e));
		out.push(index(Closed) + "," + index(Open) + "," + same(Closed, Closed) + "," + same(Closed, Waiting));
		final arr:Array<Null<Phase>> = [Closed, null, Waiting];
		out.push((arr[0] == null) + "/" + (arr[1] == null) + "/" + (arr[2] == null));
		final m = new Map<Int, Phase>();
		m.set(1, Closed);
		out.push((m.get(1) == null) + "," + (m.get(2) == null));
		final f:(Phase, Int) -> Null<Phase> = (s, i) -> i == 0 ? Closed : (i == 1 ? null : s);
		out.push((f(Open, 0) != null) + "," + Std.string(f(Open, 0)) + "," + (f(Open, 1) == null));
		final machine = new Machine<Phase, Signal>(Closed, (s, e) -> switch [s, e] {
			case [Closed, Rest]: Waiting;
			case [Waiting, Timeout]: Open;
			case [Waiting, Leave] | [Open, Leave]: Closed;
			case _: null;
		});
		machine.send(Rest);
		machine.send(Leave);
		machine.send(Rest);
		machine.send(Timeout);
		machine.send(Rest);
		out.push(Std.string(machine.state.get()) + " " + machine.log.join(" "));
		final got = out.join(" | ");
		final want = "false,false,true | Closed,Waiting,Open | true,Closed,true | false,Closed,Closed"
			+ " | Closed,false,true | 0,2,true,false | false/true/false | false,true | true,Closed,true"
			+ " | Open Rest:Closed>Waiting Leave:Waiting>Closed Rest:Closed>Waiting Timeout:Waiting>Open Rest:Open>null";
		if (got != want)
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
