// Indexing a Null<Array<Float>> reads the element as the array's own type.
class NullArrayIndex {
	static function points(full:Bool):Null<Array<Float>> {
		return full ? [0, 0, 0.2, 1] : null;
	}

	static function main() {
		var p = points(true);
		var sum = switch points(true) {
			case null: -1.0;
			case q: q[0] + q[1] + q[2] + q[3];
		}
		var s:Null<Array<Float>> = [0.5, 1.5];
		var got = p[2] + "," + sum + "," + s[1] + "," + (points(false) == null);
		if (got != "0.2,1.2,1.5,true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
