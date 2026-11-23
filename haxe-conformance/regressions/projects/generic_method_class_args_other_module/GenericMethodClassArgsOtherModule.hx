import lib.Machine;
import lib.Tip;

// A method of a generic class from another module, naming only its second
// type parameter, is specialized with both in the class's order, also when
// called on `this` and from a closure inside the class.
class GenericMethodClassArgsOtherModule {
	static function main() {
		final m = new Machine<TipState, TipEvent>(Closed, (s, e) -> switch [s, e] {
			case [Closed, Rest]: Waiting;
			case [Waiting, Timeout]: Open;
			case [Waiting, Leave] | [Open, Leave]: Closed;
			case _: null;
		});
		m.send(Rest);
		m.relay(Timeout);
		m.send(Rest);
		m.later(Leave);
		final got = Std.string(m.current()) + " " + m.log.join(" ");
		if (got != "Closed Rest:Closed>Waiting Timeout:Waiting>Open Rest:Open>null Leave:Open>Closed")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
