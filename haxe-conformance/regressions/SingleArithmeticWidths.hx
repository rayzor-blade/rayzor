// Single values combined into a Float keep one width per operation, as text
// layout computes a baseline from a font's metrics read as Single.
#if rayzor
private typedef F32 = Single;
#else
private typedef F32 = Float;
#end

abstract Metrics(haxe.io.Bytes) from haxe.io.Bytes {
	public inline function getF32(pos:Int):F32
		return cast this.getFloat(pos);
}

class Bounds {
	public final y:F32;

	public function new(y:F32)
		this.y = y;
}

class Run {
	public var above:Float = 0;
	public var below:Float = 0;

	public function new() {}
}

class SingleArithmeticWidths {
	final info:Metrics;

	function new() {
		var b = haxe.io.Bytes.alloc(16);
		b.setFloat(0, 24.0);
		b.setFloat(8, 18.5);
		b.setFloat(12, -4.5);
		info = b;
	}

	function shape(run:Run, n:Int) {
		if (n > 0) {
			var lineHeight = info.getF32(0),
				ascender = info.getF32(8),
				descender = info.getF32(12);
			run.above = (lineHeight - (ascender - descender)) / 2 + ascender;
			run.below = lineHeight - run.above;
		}
	}

	function baselineOf(b:Null<Bounds>, a:Null<Bounds>):Float {
		for (i in 0...2) {
			if (b == null || a == null)
				break;
			var lineHeight = info.getF32(0),
				ascender = info.getF32(8),
				descender = info.getF32(12);
			return b.y - a.y + (lineHeight - (ascender - descender)) / 2 + ascender;
		}
		return -1;
	}

	static function main() {
		var f = new SingleArithmeticWidths();
		var run = new Run();
		f.shape(run, 1);
		var target:F32 = 0.5;
		var start:F32 = 1.5;
		var mixed = start + (target - start) * 0.5;
		var got = [
			run.above, run.below, f.baselineOf(new Bounds(10.0), new Bounds(4.0)), f.baselineOf(null, null), mixed
		].join(" ");
		if (got != "19 5 25 -1 1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
