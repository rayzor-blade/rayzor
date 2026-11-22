// A variable a closure captures, read inside a catch, is not taken as its
// value on the path that did not throw.
typedef Ctx = {final fontSize:Float; final ?glass:Map<String, String>;};

class Deco {
	public final lines:Int;
	public final style:Null<Int>;
	public final color:Null<Int>;
	public final thickness:Null<Float>;
	public final offset:Null<Float>;

	public function new(lines:Int, style:Null<Int>, color:Null<Int>, thickness:Null<Float>, offset:Null<Float>) {
		this.lines = lines;
		this.style = style;
		this.color = color;
		this.thickness = thickness;
		this.offset = offset;
	}

	public function toString()
		return '$lines $style $color $thickness $offset';
}

class CapturedReadInCatch {
	static function shorthand(v:String, c:Ctx):{lines:Null<Int>, style:Null<Int>, colorSet:Bool, color:Null<Int>, thicknessSet:Bool, thickness:Null<Float>} {
		if (v == "bad")
			throw "bad shorthand";
		return {lines: v.length, style: null, colorSet: v.length > 3, color: 0xff0000, thicknessSet: true, thickness: c.fontSize / 8};
	}

	static function linesOf(v:String):Int {
		if (v == "blink")
			throw 'text-decoration-line $v is not supported';
		return v == "underline" ? 1 : 2;
	}

	static function styleOf(v:String):Int {
		if (v == "wavy")
			throw "wavy";
		return v.length;
	}

	static function colorOf(v:String, c:Ctx):Null<Int> {
		if (v == "currentcolor")
			return null;
		if (v == "junk")
			throw "junk colour";
		return v.length * 1000;
	}

	static function lengthOf(v:String, c:Ctx):Null<Float> {
		if (v == "auto")
			return null;
		if (v.charAt(v.length - 1) == "%")
			return Std.parseFloat(v) / 100 * c.fontSize;
		var f = Std.parseFloat(v);
		if (Math.isNaN(f))
			throw 'expected a length, not "$v"';
		return f;
	}

	static function decoration(name:String, v:String, c:Ctx):Null<Deco> {
		var declared = c.glass;
		inline function of(wanted:String):Null<String>
			return name == wanted ? v : (declared == null ? null : declared.get(wanted));

		var lines:Null<Int> = null;
		var style:Null<Int> = null;
		var color:Null<Int> = null, hasColor = false;
		var thickness:Null<Float> = null, hasThickness = false;

		var sh = of("text-decoration");
		if (sh != null)
			try {
				var p = shorthand(sh, c);
				lines = p.lines == null ? 0 : p.lines;
				style = p.style;
				if (p.colorSet) {
					hasColor = true;
					color = p.color;
				}
				if (p.thicknessSet) {
					hasThickness = true;
					thickness = p.thickness;
				}
			} catch (e:String) {
				if (name == "text-decoration")
					throw e;
			}
		var line = of("text-decoration-line");
		if (line != null)
			try
				lines = linesOf(line)
			catch (e:String) {
				if (name == "text-decoration-line")
					throw e;
			}
		var s = of("text-decoration-style");
		if (s != null)
			try
				style = styleOf(s)
			catch (e:String) {
				if (name == "text-decoration-style")
					throw e;
			}
		var col = of("text-decoration-color");
		if (col != null)
			try {
				color = colorOf(col, c);
				hasColor = true;
			} catch (e:String) {
				if (name == "text-decoration-color")
					throw e;
			}		if (lines == null)
			return null;
		return new Deco(lines, style, hasColor ? color : null, hasThickness ? thickness : null, null);
	}

	static function main() {
		var glass = ["text-decoration-color" => "junk", "text-decoration-style" => "wavy"];
		var c:Ctx = {fontSize: 16.0, glass: glass};
		var got = [
			Std.string(decoration("text-decoration", "underline", c)),
			Std.string(decoration("text-decoration-line", "underline", c)),
			Std.string(decoration("text-decoration-style", "dotted", {fontSize: 10.0})),
			Std.string(decoration("x", "q", {fontSize: 10.0, glass: ["text-decoration" => "bad", "text-decoration-line" => "overline"]}))
		];
		var thrown = try Std.string(decoration("text-decoration-color", "junk", c)) catch (e:String) "threw " + e;
		got.push(thrown);
		var want = "9 null 16711680 2 null|1 null null null null|null|2 null null null null|threw junk colour";
		if (got.join("|") != want)
			throw got.join("|");
		Sys.println("CONFORMANCE_OK");
	}
}
