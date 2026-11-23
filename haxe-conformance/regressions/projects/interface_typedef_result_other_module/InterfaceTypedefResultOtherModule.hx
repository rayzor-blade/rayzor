import lib.Highlighter;
import lib.Words;

class InterfaceTypedefResultOtherModule {
	static function main() {
		var h:Highlighter = new Words();
		var result = h.highlight("static Void trace", "s");
		var kinds = [for (token in result.tokens) token.kind + ":" + (token.to - token.from)];
		var m:Measurer = new Words();
		var extent = m.measure("a bc def");
		var got = [
			Std.string(result.tokens.length),
			kinds.join(","),
			result.state,
			Std.string(h.highlight("", "").tokens.length),
			extent.width + "/" + extent.parts.length + "/" + extent.parts[2]
		].join(" ");
		if (got != "3 plain:6,type:4,plain:5 s+ 0 8/3/def")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
