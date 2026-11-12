// `continue` in a key => value loop advances to the next entry, over an array
// and over a map, and carries values changed before it.
class KeyValueLoopContinue {
	static function main() {
		var arr = ["a", "b", "c", "d"];
		var out:Array<String> = [];
		for (i => s in arr) {
			if (s == "b")
				continue;
			out.push(i + s);
		}
		var seen = 0;
		for (i => s in arr) {
			seen += i;
			if (s == "c")
				continue;
			out.push(s);
		}
		var m = [1 => "a", 2 => "b", 3 => "c"];
		var fromMap:Array<String> = [];
		var total = 0;
		for (k => v in m) {
			total += k;
			if (v == "b")
				continue;
			fromMap.push(k + v);
		}
		fromMap.sort((a, b) -> a < b ? -1 : 1);
		var nested = 0;
		for (i => _ in arr) {
			for (j => t in arr) {
				if (j <= i)
					continue;
				if (t == "d")
					continue;
				nested++;
			}
		}
		var got = out.join(",") + "|" + seen + "|" + fromMap.join(",") + "|" + total + "|" + nested;
		if (got != "0a,2c,3d,a,b,d|6|1a,3c|6|3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
