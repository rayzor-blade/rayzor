// A local placed into an array, object or map literal moves into it: it is
// not freed when its scope ends, so later allocations cannot reuse it.
class LocalIntoLiteralLabel {
	public var v:Int;
	public var tag:String;

	public function new(v:Int, tag:String) {
		this.v = v;
		this.tag = tag;
	}
}

class LocalIntoLiteralBox {
	public var kids:Array<LocalIntoLiteralLabel>;

	public function new(kids:Array<LocalIntoLiteralLabel>) {
		this.kids = kids;
	}
}

class LocalIntoLiteralNotFreed {
	static function main() {
		var cells:Array<LocalIntoLiteralBox> = [];
		var anons:Array<{l:LocalIntoLiteralLabel}> = [];
		var maps:Array<Map<String, LocalIntoLiteralLabel>> = [];
		for (k in 0...3) {
			var label = new LocalIntoLiteralLabel(k, "a" + k);
			cells.push(new LocalIntoLiteralBox([label]));
			var inAnon = new LocalIntoLiteralLabel(k + 10, "b" + k);
			anons.push({l: inAnon});
			var inMap = new LocalIntoLiteralLabel(k + 20, "c" + k);
			maps.push(["x" => inMap]);
		}
		var churn = [for (i in 0...64) new LocalIntoLiteralLabel(-1, "z")];
		var got = [for (k in 0...3) cells[k].kids[0].v + cells[k].kids[0].tag + anons[k].l.v + maps[k]["x"].tag].join(",");
		if (got != "0a010c0,1a111c1,2a212c2" || churn.length != 64)
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
