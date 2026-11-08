import b.M;

class Main {
	static function main() {
		if (M.four(20.5, 1, 2, 3) != 26.5) throw "four";
		if (M.five(20.5, 1, 2, 3, 4) != 30.5) throw "five";
		if (M.seven(20.5, 1.5, 1, 2, 3, 4, 5) != 37.0) throw "seven";
		if (M.nine(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5) != 49.5) throw "nine";
		var m = new M();
		if (m.sized("t", 16.5, 1.25, 1, 2, 3) != 23.75) throw "instance";
		Sys.println("CONFORMANCE_OK");
	}
}
