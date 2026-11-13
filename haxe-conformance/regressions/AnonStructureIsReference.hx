// A structure is a reference: a write through any variable, loop variable or
// array element holding it is seen through all of them.
class AnonStructureIsReference {
	static function place(shelves:Array<{x:Int, height:Int}>, width:Int):Int {
		for (shelf in shelves)
			if (shelf.x + width <= 100) {
				var at = shelf.x;
				shelf.x += width;
				return at;
			}
		return -1;
	}

	static function main() {
		var a = {n: 1};
		var b = a;
		b.n = 5;
		var c:{n:Int};
		c = a;
		c.n += 1;
		var arr = [{n: 1}, {n: 2}];
		var e = arr[0];
		e.n = 9;
		for (item in arr)
			item.n += 10;
		var shelves = [{x: 25, height: 24}];
		var spots = [for (_ in 0...3) place(shelves, 25)];
		var got = [a.n, b.n, arr[0].n, arr[1].n].join(",") + " " + spots.join(",") + " " + shelves[0].x;
		if (got != "6,6,19,12 25,50,75 100")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
