// A null class reference upcast to an interface, or to a nullable abstract
// over one, stays null; a non-null one dispatches through the interface.
interface NullClassToInterfaceSig<T> {
	function get():T;
}

class NullClassToInterfaceDyn<T> implements NullClassToInterfaceSig<T> {
	var v:T;

	public function new(v:T) {
		this.v = v;
	}

	public function get():T {
		return v;
	}
}

@:forward(get)
abstract NullClassToInterfaceBox<T>(NullClassToInterfaceSig<T>) from NullClassToInterfaceSig<T> to NullClassToInterfaceSig<T> {}

class NullClassToInterface {
	static var held:Null<NullClassToInterfaceBox<Array<String>>> = null;

	static function make(full:Bool):NullClassToInterfaceDyn<Int> {
		return full ? new NullClassToInterfaceDyn(1) : null;
	}

	static function pick(full:Bool):Null<NullClassToInterfaceBox<Array<String>>> {
		return full ? new NullClassToInterfaceDyn(["x"]) : null;
	}

	static function main() {
		var out = [];
		var c = make(false);
		var i:NullClassToInterfaceSig<Int> = c;
		out.push(i == null ? "null" : "set");
		var j:NullClassToInterfaceSig<Int> = make(true);
		out.push("" + j.get());
		out.push(held == null ? "null" : "set");
		held = new NullClassToInterfaceDyn(["a", "b"]);
		out.push("" + held.get().length);
		held = pick(false);
		out.push(held == null ? "null" : "set");
		held = pick(true);
		out.push("" + held.get().length);
		var got = out.join(",");
		if (got != "null,1,null,2,null,1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
