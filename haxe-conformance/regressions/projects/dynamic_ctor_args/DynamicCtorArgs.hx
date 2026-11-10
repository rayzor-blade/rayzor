import b.Cell;

class DynamicCtorArgs {
	static function check(got:Dynamic, want:String) {
		if ("" + got != want)
			throw got + " != " + want;
	}

	static function main() {
		var half:Single = 0.5;
		check(new Cell<Dynamic>(half).get(), "0.5");
		check(new Cell<Dynamic>(2.5).get(), "2.5");
		check(new Cell<Dynamic>(3).get(), "3");
		check(new Cell<Dynamic>(true).get(), "true");
		check(new Cell<Dynamic>("s").get(), "s");
		check(new Tagged<Int, Dynamic>(half).get(), "0.5");
		check(new Tagged<Int, Dynamic>(1.5).get(), "1.5");
		Sys.println("CONFORMANCE_OK");
	}
}
