// A generic call's function literal takes its parameter types from what an
// earlier literal argument bound, as Haxe infers them.
class LambdaParameterFromEarlierLambda {
	static var out:Array<String> = [];

	static function call<T>(work:Void->T, finished:T->Void):Void
		finished(work());

	static function main() {
		call(() -> 2 * 3, v -> out.push("int " + (v + 1)));
		call(() -> "abc".toUpperCase(), v -> out.push("len " + v.length + " " + v.toLowerCase() + " " + v));
		call(() -> "ab" + "c".toUpperCase(), v -> out.push(v));
		var got = out.join(",");
		if (got != "int 7,len 3 abc ABC,abC")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
