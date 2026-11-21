// A Float or Bool result of an interface method returning its type parameter,
// read where the parameter is bound to Null<Float> or Null<Bool>, is that value.
interface InterfaceNullableScalarResultIC<T> {
	function get():T;
}

class InterfaceNullableScalarResultF implements InterfaceNullableScalarResultIC<Float> {
	public function new() {}

	public function get():Float
		return 0.2;
}

class InterfaceNullableScalarResultB implements InterfaceNullableScalarResultIC<Bool> {
	public function new() {}

	public function get():Bool
		return true;
}

class InterfaceNullableScalarResult {
	static function main() {
		var min:Null<Float> = 0.0;
		var f:InterfaceNullableScalarResultIC<Null<Float>> = cast new InterfaceNullableScalarResultF();
		var g = f.get();
		var scaled = (g - min) / 2;
		var b:InterfaceNullableScalarResultIC<Null<Bool>> = cast new InterfaceNullableScalarResultB();
		var on = b.get();
		var got = g + " " + scaled + " " + (on == true) + " " + (g != null);
		if (got != "0.2 0.1 true true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
