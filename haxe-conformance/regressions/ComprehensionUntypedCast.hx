// An untyped `cast` in a comprehension the context types as Array<T> is a
// cast to T: the elements are the objects themselves.
class CastBase {
	public function new() {}
}

class CastLeaf extends CastBase {
	public var w:String;

	public function new(w:String) {
		super();
		this.w = w;
	}
}

class ComprehensionUntypedCast {
	static function leaves(items:Array<CastBase>):Array<CastLeaf>
		return [for (c in items) if (Std.isOfType(c, CastLeaf)) cast c];

	static function main() {
		var a = new CastLeaf("60"), b = new CastLeaf("2*");
		var items:Array<CastBase> = [a, new CastBase(), b];
		var held:Array<CastLeaf> = [for (c in items) if (Std.isOfType(c, CastLeaf)) cast c];
		var back = leaves(items);
		var got = [held[0] == a, held[1] == b, back[1] == b].join(",") + " " + held[0].w + " " + back[1].w;
		if (got != "true,true,true 60 2*")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
