import s.Style;

typedef Handler = (String, Int) -> Int;

class Main {
	static function build():Map<String, Handler> {
		var h = new Map<String, Handler>();
		var aligns = ["start" => Align.Start, "end" => Align.End, "normal" => Align.Auto];
		h.set("items", (n, v) -> (aligns.get(n) : Int) + v);
		var selfAligns = aligns.copy();
		selfAligns.set("auto", Align.Auto);
		h.set("self", (n, v) -> (selfAligns.get(n) : Int) + v);
		return h;
	}

	static function main() {
		var a = Align.End;
		if ((a : Int) != 1) throw "imported Align.End";
		if (Flow_left() != 0 || t.Flow.center() != 1) throw "private Align inside its module";
		var h = build();
		if (h.get("items")("end", 10) != 11) throw "captured map";
		if (h.get("self")("auto", 10) != 9) throw "captured copy";
		Sys.println("CONFORMANCE_OK");
	}

	static function Flow_left():Int
		return t.Flow.left();
}
