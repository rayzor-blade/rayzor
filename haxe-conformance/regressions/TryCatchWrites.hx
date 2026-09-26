// A catch, and the code after a try, see what the try body wrote before it
// threw -- whether the throw is direct or from a called function, in the
// statement and the expression form of try.
class TryCatchWrites {
	static function fail() throw "x";

	static function main() {
		var out = [];
		var x = 0;
		try {
			x = 10;
			throw "";
		} catch (_) {
			x += 1;
		}
		out.push(x);
		var y = 0;
		try {
			y = 10;
			fail();
		} catch (e:Dynamic) {}
		out.push(y);
		var z = 1;
		var r = try {
			z = 5;
			fail();
			0;
		} catch (e:Dynamic) z * 2;
		out.push(r);
		var n = 0;
		for (i in 0...3) {
			try {
				n += i;
				if (i == 1) throw "skip";
				n += 100;
			} catch (e:String) {}
		}
		out.push(n);
		var got = out.join(" ");
		if (got == "11 10 10 203")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
