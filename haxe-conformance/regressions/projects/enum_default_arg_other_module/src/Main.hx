class Main {
	static function main() {
		var got = [p.Pointer.press(1), p.Pointer.press(2, Right), p.Pointer.second(3), p.Pointer.press(4, Other(7)), p.Pointer.label(5)].join(",");
		if (got != "1 Left true,2 Right false,3 Right,4 Other(7) false,5 none")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
