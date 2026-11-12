// An untyped lambda passed to a generic class's constructor takes its
// parameter types from the class type parameter another argument fixes.
class CtorLambdaFromTypeParamWatch<T> {
	public function new(read:Void->T, react:T->Void) {
		react(read());
	}
}

class CtorLambdaFromTypeParamGen<T> {
	public var seen = "";

	public function new(each:Void->Array<T>) {
		new CtorLambdaFromTypeParamWatch(each, list -> seen = "rows " + list.length);
	}
}

class CtorLambdaFromTypeParam {
	static function main() {
		var out = [];
		new CtorLambdaFromTypeParamWatch(() -> [0, 1], list -> out.push("len " + list.length));
		new CtorLambdaFromTypeParamWatch(() -> "abc", s -> out.push("str " + s.length));
		new CtorLambdaFromTypeParamWatch(() -> 2.5, f -> out.push("half " + f / 2));
		out.push(new CtorLambdaFromTypeParamGen(() -> ["a", "b", "c"]).seen);
		var got = out.join(",");
		if (got != "len 2,str 3,half 1.25,rows 3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
