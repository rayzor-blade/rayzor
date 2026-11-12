package lib;

enum Kind {
	A;
	B;
}

class Reg {
	static final byId = new Map<Int, Reg>();
	static final byObj = new Map<Reg, String>();
	static final byKind = new Map<Kind, Int>();

	public final id:Int;

	function new(id:Int)
		this.id = id;

	public static function make(id:Int):Reg {
		var r = byId.get(id);
		if (r != null)
			return r;
		r = new Reg(id);
		byId.set(id, r);
		byObj.set(r, "r" + id);
		byKind.set(id % 2 == 0 ? A : B, id);
		return r;
	}

	public static function describe(r:Reg):String
		return byObj.get(r) + " " + byKind.get(A) + " " + byKind.get(B);
}
