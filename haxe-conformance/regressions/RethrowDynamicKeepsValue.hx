// `catch (e:Dynamic) { throw e; }` rethrows the value itself, caught again by
// its own type; a thrown Float keeps its fraction.
class RethrowDynamicOops {
	public var code:Int;

	public function new(c)
		code = c;
}

class RethrowDynamicKeepsValue {
	static function rethrow(f:Void->Void) {
		try {
			f();
		} catch (e:Dynamic) {
			throw e;
		}
	}

	static function main() {
		var out = [];
		try rethrow(() -> throw "s") catch (s:String) out.push("str " + s);
		try rethrow(() -> throw 7) catch (i:Int) out.push("int " + i);
		try rethrow(() -> throw 2.5) catch (f:Float) out.push("float " + f);
		try rethrow(() -> throw new RethrowDynamicOops(3)) catch (o:RethrowDynamicOops) out.push("oops " + o.code);
		try rethrow(() -> throw true) catch (b:Bool) out.push("bool " + b);
		try rethrow(() -> throw "gpu error") catch (ex:haxe.Exception) out.push(ex.message);
		try throw 1.25 catch (d:Dynamic) out.push("dyn " + d);
		var got = out.join(", ");
		if (got != "str s, int 7, float 2.5, oops 3, bool true, gpu error, dyn 1.25")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
