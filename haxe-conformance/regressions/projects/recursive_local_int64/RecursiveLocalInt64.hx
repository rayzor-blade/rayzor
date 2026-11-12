// A local function that calls itself with an Int64 (through its own closure
// value) receives the value, not the box's address.
import haxe.Int64;

class RecursiveLocalInt64 {
	static function main() {
		var out = [];
		function walk(id:Int64, depth:Int) {
			out.push(Int64.toStr(id));
			if (depth < 2)
				for (k in Tree.children(id))
					walk(k, depth + 1);
		}
		walk(Int64.make(1, 8), 0);
		var got = out.join(" ");
		if (got != "4294967304 4294967305 4294967306 4294967307 4294967306 4294967307 4294967308")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
