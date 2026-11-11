// A block whose value is a local it declares hands that local's object out, not a freed one.
class Item {
	public final n:Int;

	public function new(n:Int)
		this.n = n;
}

class Holder {
	public final items:Array<Item>;

	public function new(items:Array<Item>)
		this.items = items;
}

class BlockValueLocal {
	static function sum(items:Array<Item>):Int {
		var t = 0;
		for (i in items)
			t += i.n;
		return t;
	}

	static function main() {
		var a = [{
			var t = new Item(3);
			t;
		}];
		var s = sum([{
			var t = new Item(4);
			t;
		}]);
		var c = {
			var t = new Item(6);
			t;
		};
		var h = new Holder([{
			var t = new Item(8);
			t;
		}]);
		var got = [a[0].n, s, c.n, h.items[0].n].join(",");
		if (got != "3,4,6,8")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
