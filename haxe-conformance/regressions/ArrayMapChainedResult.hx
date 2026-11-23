// `Array.map` takes its result element type from what the callback returns,
// so a map over a map's result reads the values the first callback produced.
enum Phase {
	Closed;
	Opening;
}

class ArrayMapChainedResult {
	static function main() {
		final out = [];
		final ints = [1, 2, 3];
		final nullable:Array<Null<Int>> = [0, null, 2];
		out.push(nullable.map(x -> x).map(x -> x == 0).join(","));
		out.push(ints.map(x -> x * 2).map(x -> x + 1).join(","));
		out.push(ints.map(x -> "s" + x).map(x -> x.length).join(","));
		final strings = ints.map(x -> "s" + x);
		out.push(strings.map(x -> x.length).join(","));
		out.push(ints.map(x -> x * 0.5).map(x -> x + 1).join(","));
		out.push(ints.map(x -> x > 1).map(x -> !x).join(","));
		final phases:Array<Null<Phase>> = [Closed, null, Opening];
		out.push(phases.map(x -> x).map(x -> x == Closed).join(","));
		out.push(ints.map(x -> [x, x]).map(x -> x.length).join(","));
		out.push(ints.map(x -> {v: x}).map(x -> x.v + 1).join(","));
		out.push(ints.map(x -> x > 1 ? "a" : null).map(x -> x == null).join(","));
		out.push(["a", "bb"].map(x -> x + "!").join(","));
		final got = out.join(" | ");
		final want = "true,false,false | 3,5,7 | 2,2,2 | 2,2,2 | 1.5,2,2.5 | true,false,false"
			+ " | true,false,false | 2,2,2 | 2,3,4 | true,false,false | a!,bb!";
		if (got != want)
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
