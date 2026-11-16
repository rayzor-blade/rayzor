import lib.Watch;

class GenericClosureFieldsOtherModule {
	static function main() {
		var flag = false;
		var count = 10;
		var out:Array<String> = [];
		new Watch(() -> flag, v -> out.push(v ? "T" : "F"));
		new Watch(() -> count, v -> out.push("n" + v));
		new Watch(() -> count * 2, v -> out.push("d" + v), (a, b) -> a == b);
		flag = true;
		count = 20;
		Watch.flush();
		var got = out.join(",");
		if (got != "F,n10,d20,T,n20,d40")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
