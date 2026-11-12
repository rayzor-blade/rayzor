// Null<Float> values in structures and arrays keep their value and their
// null: a literal's field, an element stored by index (zero too), and a field
// read through a structure of plain Float built by a comprehension.
typedef NullFloatStop = {
	final color:Int;
	final offset:Null<Float>;
}

class NullFloatStructureFields {
	static function placed(stops:Array<NullFloatStop>):Array<{color:Int, offset:Float}> {
		var offsets:Array<Null<Float>> = [for (s in stops) s.offset];
		if (offsets[0] == null)
			offsets[0] = 0;
		if (offsets[offsets.length - 1] == null)
			offsets[offsets.length - 1] = 1;
		var i = 0;
		while (i < offsets.length) {
			if (offsets[i] != null) {
				i++;
				continue;
			}
			var start = i - 1, end = i;
			while (offsets[end] == null)
				end++;
			for (k in i...end)
				offsets[k] = offsets[start] + (offsets[end] - offsets[start]) * (k - start) / (end - start);
			i = end;
		}
		return [for (k in 0...stops.length) {color: stops[k].color, offset: offsets[k]}];
	}

	static function show(stops:Array<NullFloatStop>):String {
		return [for (p in placed(stops)) p.color + "@" + p.offset].join(" ");
	}

	static function main() {
		var out:Array<String> = [];
		var s:NullFloatStop = {color: 1, offset: 0.25};
		var f:Float = s.offset;
		out.push(s.offset + "," + f + "," + (s.offset + 1));
		var a:Array<Null<Float>> = [0.5, null];
		a[1] = 0;
		out.push(a[1] + " " + (a[1] == null));
		var n:Null<Float> = 0.5;
		var o:{offset:Float} = {offset: n};
		var list:Array<{offset:Float}> = [{offset: n}];
		out.push(o.offset + "," + list[0].offset);
		out.push(show([{color: 1, offset: null}, {color: 2, offset: null}, {color: 3, offset: null}]));
		out.push(show([{color: 1, offset: 0.25}, {color: 2, offset: 0.75}]));
		var got = out.join("|");
		if (got != "0.25,0.25,1.25|0 false|0.5,0.5|1@0 2@0.5 3@1|1@0.25 2@0.75")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
