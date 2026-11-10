// Growing an empty array literal through every growing operation.
class ArrayGrowFromEmpty {
	static function main() {
		var total = 0;
		for (i in 0...200) {
			var a:Array<Int> = [];
			a.insert(0, 10);
			a.insert(1, 11);
			a.insert(2, 16);
			a.insert(0, 9);
			total += a.length;
			if (a.join(",") != "9,10,11,16")
				throw a.join(",");
		}
		if (total != 800)
			throw "total " + total;
		var f:Array<Float> = [];
		for (i in 0...20)
			f.insert(0, i + 0.5);
		if (f.length != 20 || f[0] != 19.5 || f[19] != 0.5)
			throw "float " + f;
		var s:Array<String> = [];
		for (i in 0...20)
			s.insert(s.length, "s" + i);
		if (s.join("") != [for (i in 0...20) "s" + i].join(""))
			throw "string " + s;
		var u:Array<Int> = [];
		for (i in 0...20)
			u.unshift(i);
		if (u[0] != 19 || u[19] != 0)
			throw "unshift " + u;
		var g:Array<Int> = [];
		g[12] = 5;
		if (g.length != 13 || g[12] != 5)
			throw "set " + g.length;
		var r:Array<Int> = [];
		r.resize(30);
		r[29] = 1;
		if (r.length != 30 || r[29] != 1)
			throw "resize " + r.length;
		Sys.println("CONFORMANCE_OK");
	}
}
