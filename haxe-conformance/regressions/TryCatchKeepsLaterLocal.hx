// An object read after a try/catch is still alive there, whether the try is a
// statement or an expression and whether the object is used inside it.
class TryCatchHolder {
	public var m:String;

	public function new(m:String)
		this.m = m;
}

class TryCatchKeepsLaterLocal {
	static function main() {
		var out = [];
		var a = new TryCatchHolder("a");
		try
			throw "t"
		catch (x:String) {}
		var got = a.m;
		out.push(got);

		var b = new TryCatchHolder("b");
		var v = try {
			out.push("in " + b.m);
			throw 1;
		} catch (n:Int) n;
		var after = b.m + v;
		out.push(after);

		var c = new TryCatchHolder("c");
		try {
			out.push("try");
		} catch (x:String) {} finally {
			out.push("finally");
		}
		var last = c.m;
		out.push(last);

		var joined = out.join(",");
		if (joined != "a,in b,b1,try,finally,c")
			throw joined;
		Sys.println("CONFORMANCE_OK");
	}
}
