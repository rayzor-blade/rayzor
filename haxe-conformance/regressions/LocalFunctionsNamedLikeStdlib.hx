// A local function whose name a stdlib static shares (sys.io.File.append,
// File.write) is called through its own value, never bound to the stdlib
// function by name.
class LocalFunctionsNamedLikeStdlib {
	static function main() {
		var buf = new StringBuf();
		function append(v:Int) buf.add(Std.string(v));
		var write = function(s:String) buf.add(s);
		for (i in 0...3) {
			switch i {
				case n if (n < 2):
					append(n);
				case _:
					write("!");
			}
		}
		var total = 0;
		var append2 = (v:Int) -> total += v;
		append2(5);
		if (buf.toString() == "01!" && total == 5)
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + buf.toString() + " " + total);
	}
}
