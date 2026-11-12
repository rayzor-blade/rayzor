// A structure literal assigned to a field or a static has the target's field
// types: an Int for a Float field is that Float.
typedef LiteralAssignedPt = {final x:Float; final y:Float;}

class LiteralAssignedHolder {
	public var p:LiteralAssignedPt;
	public static var s:LiteralAssignedPt;
	public var list:Array<Float>;

	public function new() {
		p = {x: 60, y: 0};
		list = [1, 2];
	}
}

class LiteralAssignedToField {
	static function main() {
		var h = new LiteralAssignedHolder();
		LiteralAssignedHolder.s = {x: 60, y: 1};
		var got = h.p.x + " " + h.p.y + " " + LiteralAssignedHolder.s.x + " " + LiteralAssignedHolder.s.y + " " + (h.list[0] * 0.5);
		if (got != "60 0 60 1 0.5")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
