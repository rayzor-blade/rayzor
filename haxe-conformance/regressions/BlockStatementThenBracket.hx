// A switch, if or loop statement ends at its `}`: a `[` or `(` on the next line starts a new statement.
class BlockStatementThenBracket {
	static function main() {
		var log = [];
		var viaSwitch = (v:Int) -> {
			switch v {
				case 1:
					log.push("one");
				case _:
					log.push("other");
			}
			[v * 10];
		};
		var viaFor = (n:Int) -> {
			for (i in 0...n) {
				log.push("i" + i);
			}
			[n];
		};
		function viaWhile(n:Int):Array<Int> {
			var k = 0;
			while (k < n) {
				k++;
			}
			return [k];
		}
		var got = [viaSwitch(1)[0], viaSwitch(2)[0], viaFor(2)[0], viaWhile(3)[0]].join(",") + "|" + log.join(",");
		if (got != "10,20,2,3|one,other,i0,i1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
