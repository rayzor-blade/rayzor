// A clause that takes every exception ends the chain: the ones after it never run.
class CatchAfterCatchAll {
	static function run(f:Void->Void):String {
		var out = [];
		try {
			f();
		} catch (e:haxe.Exception) {
			out.push("E:" + e.message);
		} catch (e:Dynamic) {
			out.push("D");
		}
		try {
			f();
		} catch (e:String) {
			out.push("S:" + e);
		} catch (e:Dynamic) {
			out.push("D2");
		}
		return out.join(",");
	}

	static function main() {
		var got = run(() -> throw "str") + " " + run(() -> throw new haxe.Exception("ex"));
		if (got != "E:str,S:str E:ex,D2")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
