// A root-package module's own typedef names that typedef, though an imported
// module declares a private enum of the same name; and an empty array literal
// beside an array in a conditional takes that array's element type.
import other.Flow;

typedef Item = {
	label:String
}

class Box<T> {
	public var v:T;

	public function new(v:T)
		this.v = v;

	public function get():T
		return v;
}

class OwnTypedefOverImportedPrivateEnum {
	static function main() {
		var items:Array<Item> = [{label: "first"}, {label: "second"}];
		var a:Item = {label: "direct"};
		var b = new Box<Item>({label: "boxed"});
		var pick = () -> {
			var i = 1;
			i >= 0 && i < items.length ? items[i] : null;
		};
		var lines = (t:String) -> t == "" ? [] : t.split(",");
		var got = [a.label, b.get().label, pick().label, lines("x,y").join("|"), Std.string(lines("").length), Std.string(Flow.first())].join(" ");
		if (got != "direct boxed second x|y 0 Word(w)")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
