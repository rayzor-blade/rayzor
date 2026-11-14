// A lambda passed for a `Null<E>` callback returns null as null and a variant
// as that variant, as a named function does.
enum Phase {
	Idle;
	Dragging;
	Settled;
}

enum Ev {
	Begin;
	End;
}

class LambdaNullableEnumResult {
	static function run(t:(Phase, Ev) -> Null<Phase>, s:Phase, e:Ev):String {
		var to = t(s, e);
		return to == null ? "null" : to == Dragging ? "dragging" : to == Settled ? "settled" : "other";
	}

	static function named(s:Phase, e:Ev):Null<Phase>
		return e == Begin ? Dragging : null;

	static function main() {
		var got = [
			run((s, e) -> e == Begin ? Dragging : null, Idle, End),
			run((s, e) -> e == Begin ? Dragging : null, Idle, Begin),
			run((s, e) -> switch [s, e] {
				case [Idle, Begin]: Dragging;
				case [Dragging, End]: Settled;
				case _: null;
			}, Dragging, End),
			run((s, e) -> switch [s, e] {
				case [Idle, Begin]: Dragging;
				case _: null;
			}, Settled, End),
			run(named, Idle, End),
			run(named, Idle, Begin)
		].join(",");
		if (got != "null,dragging,settled,null,null,dragging")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
