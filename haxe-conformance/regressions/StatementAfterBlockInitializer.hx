// A declaration whose initializer ends in a block, switch or if ends there:
// a parenthesised expression on the next line is its own statement.
class StatementAfterBlockInitializer {
	static function main() {
		var a = 2;
		var w = a == 0 ? 0 : switch a {
			case 1: 5;
			case _: 6;
		}
		(w + 1 : Float);
		var v = switch a {
			case 2: 7;
			case _: 8;
		}
		(v : Float);
		final x = {
			9;
		}
		(x : Float);
		var arr = [10, 11][1];
		var got = w + "," + v + "," + x + "," + arr;
		if (got != "6,7,9,11")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
