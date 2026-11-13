// A class seen through a structure with a `(get, never)` field answers with
// the property's getter.
typedef SlotView = {var slot(get, never):Int; var width(default, null):Int;}

class SlotBitmap {
	static var made = 0;

	final id:Int;

	public var slot(get, never):Int;
	public var width(default, null):Int;

	public function new(width:Int) {
		id = made++;
		this.width = width;
	}

	function get_slot():Int
		return 100 + id;
}

class StructuralGetterProperty {
	static function describe(view:SlotView):String
		return view.slot + "x" + view.width;

	static function main() {
		var a = new SlotBitmap(1), b = new SlotBitmap(2);
		var views:Array<SlotView> = [a, b];
		var got = describe(a) + " " + describe(b) + " " + [for (v in views) v.slot].join(",");
		if (got != "100x1 101x2 100,101")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
