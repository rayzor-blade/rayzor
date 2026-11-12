// A conditional with a null branch, bound for a Null<Int64>, holds the value
// once: it reads back as itself, and its null branch as null.
import haxe.Int64;

class NullInt64Conditional {
	static function take(v:Int64):String {
		return Int64.toStr(v);
	}

	static function pick(full:Bool):Null<Int64> {
		return full ? Int64.make(2, 3) : null;
	}

	static function main() {
		var on = Sys.args().length < 5;
		var b:Null<Int64> = on ? Int64.make(1, 9) : null;
		var off:Null<Int64> = !on ? Int64.make(1, 9) : null;
		var at:Null<Int64> = Int64.make(1, 8);
		var walked:Array<String> = [];
		var steps = 0;
		while (at != null && steps < 3) {
			walked.push(take(at));
			at = steps == 0 ? Int64.make(1, 10) : null;
			steps++;
		}
		var got = take(b) + "|" + (off == null) + "|" + (pick(false) == null) + "|" + take(pick(true)) + "|" + walked.join(",");
		if (got != "4294967305|true|true|8589934595|4294967304,4294967306")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
