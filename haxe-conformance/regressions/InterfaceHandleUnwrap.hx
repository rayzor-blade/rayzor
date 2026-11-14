// A class instance held as an interface, or as an abstract over one, is the
// instance itself when boxed to Dynamic or cast back to its class.
private interface IFlag<T> {
	function get():T;
}

private class Flag implements IFlag<Bool> {
	var value:Bool;

	public function new(value:Bool)
		this.value = value;

	public function get():Bool
		return value;
}

@:forward(get)
private abstract FlagView<T>(IFlag<T>) from IFlag<T> to IFlag<T> {}

class InterfaceHandleUnwrap {
	static function main() {
		var i:IFlag<Bool> = new Flag(false);
		var v:FlagView<Bool> = new Flag(false);
		var di:Dynamic = i;
		var dv:Dynamic = v;
		var back:Flag = cast i;
		var viewBack:Flag = cast v;
		var got = [
			Std.isOfType(di, Flag), (cast di : Flag).get(), Std.isOfType(dv, Flag), (cast dv : Flag).get(),
			back.get(), viewBack.get(), Std.isOfType(i, Flag)
		].join(",");
		if (got != "true,false,true,false,false,false,true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
