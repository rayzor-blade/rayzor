package lib;

// A generic class whose methods name only some of its type parameters.
class Machine<S:EnumValue, E> {
	var state:S;
	final transition:(S, E) -> Null<S>;
	final queue:Array<E> = [];

	public final log:Array<String> = [];

	public function new(initial:S, transition:(S, E) -> Null<S>) {
		state = initial;
		this.transition = transition;
	}

	public function current():S
		return state;

	public function send(event:E):Void {
		queue.push(event);
		while (queue.length > 0) {
			final e = queue.shift();
			final to = transition(state, e);
			log.push(Std.string(e) + ":" + Std.string(state) + ">" + Std.string(to));
			if (to != null)
				state = to;
		}
	}
}
