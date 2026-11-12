// An empty array takes a pushed concatenation's String type, and join prints
// an abstract element as the value it stores, a Single as its float.
abstract ArrayJoinMeters(Int) from Int to Int {}

enum abstract ArrayJoinColor(String) to String {
	var Red = "red";
	var Blue = "blue";
}

class ArrayAbstractElementJoin {
	static function encode():Array<Single>
		return [1, 18.0, 2.5];

	static function show(a:Array<Single>):String {
		var out = [];
		for (i in 0...a.length) {
			var f:Float = a[i];
			out.push("" + f);
		}
		return out.join(",");
	}

	static function main() {
		var m:Array<ArrayJoinMeters> = [3, 4];
		var c:Array<ArrayJoinColor> = [Red, Blue];
		var flags = [];
		for (v in m)
			flags.push(v + 1 > 3);
		var got = show([1.5, 2.5]) + " " + encode().join(",") + " " + m.join(",") + " " + c.join("|") + " "
			+ flags.join(",");
		if (got != "1.5,2.5 1,18,2.5 3,4 red|blue true,true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
