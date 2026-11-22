// A generic class's Array<T> at T = Dynamic keeps scalars, objects and
// Strings side by side whichever writer comes first.
class Tint {
	public final rgb:Int;

	public function new(rgb:Int)
		this.rgb = rgb;
}

class Slots<T> {
	final items:Array<T> = [];

	public function new() {}

	public function insert(at:Int, v:T):Void
		items.insert(at, v);

	public function unshift(v:T):Void
		items.unshift(v);

	public function at(i:Int):T
		return items[i];
}

class GenericArrayInsertMixedDynamic {
	static function rgb(d:Dynamic):Int
		return (d : Tint).rgb;

	static function main() {
		var c = new Tint(0x123456);
		var got = [];

		var a = new Slots<Dynamic>();
		a.insert(0, 7);
		a.insert(1, c);
		got.push(a.at(0) + " " + rgb(a.at(1)));

		var b = new Slots<Dynamic>();
		b.insert(0, c);
		b.insert(0, 0.5);
		got.push(b.at(0) + " " + rgb(b.at(1)));

		var s = new Slots<Dynamic>();
		s.insert(0, 7);
		s.insert(1, "text");
		got.push(s.at(0) + " " + s.at(1));

		var u = new Slots<Dynamic>();
		u.unshift(c);
		u.unshift(3);
		got.push(u.at(0) + " " + rgb(u.at(1)));

		var line = got.join("|");
		if (line != "7 1193046|0.5 1193046|7 text|3 1193046")
			throw line;
		Sys.println("CONFORMANCE_OK");
	}
}
