class State {
	public function new() {}
}

class Holder {
	public final create:Void->State;
	public final createInt:Void->Null<Int>;

	public function new(create:Void->State, createInt:Void->Null<Int>) {
		this.create = create;
		this.createInt = createInt;
	}

	public function viaMethod():State
		return create();

	public function viaVar():Bool {
		var r = create();
		return r == null;
	}
}

class Radii {
	public var full:Float = 9999.0;

	public function new() {}
}

class FunctionFieldValues {
	static function dyn(f:Void->Dynamic):String
		return "" + f();

	static function main() {
		var h = new Holder(() -> null, () -> null);
		var real = new Holder(() -> new State(), () -> 5);
		var radii = new Radii();
		var parts:Array<Dynamic> = [
			h.create() == null, h.viaMethod() == null, h.viaVar(), h.createInt() == null, real.create() == null,
			dyn(() -> 9999.0), dyn(() -> radii.full), dyn(() -> 7), dyn(() -> true),
		];
		var got = parts.join(",");
		if (got != "true,true,true,true,false,9999,9999,7,true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
