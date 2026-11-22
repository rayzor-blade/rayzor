// A ternary of an Int and a Single is a Single, so it keeps its value where
// a Float is stored.
#if !rayzor
typedef Single = Float;
#end

typedef Shot = {x:Float, w:Float};

class Bounds {
	public final x:Single;
	public final width:Single;

	public function new(x:Single, width:Single) {
		this.x = x;
		this.width = width;
	}
}

class SingleTernaryFloatField {
	static function bounds(ok:Bool):Null<Bounds>
		return ok ? new Bounds(360.0, 260.5) : null;

	static function twice(v:Float):Float
		return v * 2;

	static function main() {
		final b = bounds(true), nb = bounds(false);
		final s:Shot = {x: b == null ? 0 : b.x, w: b != null ? b.width : 0};
		final n:Shot = {x: nb == null ? 0 : nb.x, w: nb != null ? nb.width : 1};
		final local:Float = b == null ? 0 : b.width;
		final single:Single = b == null ? 2 : b.x;
		final list:Array<Float> = [b == null ? 0 : b.x];
		var got = [
			s.x, s.w, n.x, n.w, local, single, list[0], twice(b == null ? 0 : b.width)
		].join(" ");
		if (got != "360 260.5 0 1 260.5 360 360 521")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
