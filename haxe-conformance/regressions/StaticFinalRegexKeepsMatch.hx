// A static final regex is one object: a match through it is still there on
// the next read, as matchSub/matchedPos loops rely on.
class StaticFinalRegexKeepsMatch {
	static final NUM = ~/(-?[0-9]*\.?[0-9]+)([a-z%]*)/g;

	static function main() {
		var s = "0.3 1px";
		var values = [], parts = [];
		var last = 0;
		var guard = 0;
		while (NUM.matchSub(s, last) && guard++ < 10) {
			var p = NUM.matchedPos();
			parts.push(s.substring(last, p.pos));
			values.push(Std.parseFloat(NUM.matched(1)));
			last = p.pos + p.len;
		}
		var got = values.join(",") + "|" + parts.join(",") + "|" + guard + "|" + (NUM == NUM) + "|" + ["", "a"].toString();
		if (got != "0.3,1|, |2|true|[,a]")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
