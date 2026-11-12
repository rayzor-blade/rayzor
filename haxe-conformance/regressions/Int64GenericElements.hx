// Int64 values above 2^32 read back through a generic callback (Lambda.exists
// over an Array<Int64>) as themselves, aligned or not, next to boxed ones.
import haxe.Int64;

class Int64GenericElements {
	static function main() {
		var marked = Int64.toStr(Int64.make(1, 16));
		var ids = [Int64.make(1, 8), Int64.make(1, 16), Int64.make(2, 3)];
		var hit = Lambda.exists(ids, id -> Int64.toStr(id) == marked);
		var seen:Array<String> = [];
		Lambda.iter(ids, id -> seen.push(Int64.toStr(id)));
		var boxed:Dynamic = Int64.make(3, 40);
		var fromDynamic:Array<Int64> = [boxed];
		var back = Lambda.fold(fromDynamic, (id, acc:String) -> acc + Int64.toStr(id), "");
		var got = hit + "|" + seen.join(",") + "|" + back;
		if (got != "true|4294967304,4294967312,8589934595|12884901928")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
