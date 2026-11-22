// An Int beside a typedef of Float widens to Float, as beside Float itself,
// and so does a typedef of Int beside a Float.
typedef Px = Float;
typedef Count = Int;
typedef Shot = {x:Float, w:Float};

class Bounds {
	public final x:Px;
	public final width:Px;

	public function new(x:Px, width:Px) {
		this.x = x;
		this.width = width;
	}
}

class TypedefFloatTernary {
	static function bounds(ok:Bool):Null<Bounds>
		return ok ? new Bounds(360.0, 260.5) : null;

	static function main() {
		final b = bounds(true), nb = bounds(false);
		final s:Shot = {x: b == null ? 0 : b.x, w: b == null ? 0 : b.width};
		final n:Shot = {x: nb == null ? 0 : nb.x, w: nb != null ? nb.width : 1};
		final local:Float = b == null ? 0 : b.width;
		final list:Array<Float> = [b == null ? 0 : b.x];
		final c:Count = 3;
		final mixed:Float = b == null ? c : b.width;
		final counted:Float = b != null ? c : b.width;
		var got = [s.x, s.w, n.x, n.w, local, list[0], mixed, counted].join(" ");
		if (got != "360 260.5 0 1 260.5 360 260.5 3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
