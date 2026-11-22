// A T result of another module's generic interface, read at T = Dynamic,
// is the Bool, Int or Float the implementation holds.
import lib.Computed;

class InterfaceBoolResultOtherModule {
	static function erased<T>(r:Read<T>):Dynamic {
		var e:Read<Dynamic> = cast r;
		return e.get();
	}

	static function main() {
		var n = 3;
		var values:Array<Dynamic> = [
			erased(Make.of(() -> n == 3)),
			erased(Make.of(() -> n == 4)),
			erased(Make.of(() -> n + 1)),
			erased(Make.of(() -> n + 0.5)),
		];
		var got = [for (v in values) Std.string(v) + ":" + Std.string(Type.typeof(v))].join(" ");
		if (got != "true:TBool false:TBool 4:TInt 3.5:TFloat")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
