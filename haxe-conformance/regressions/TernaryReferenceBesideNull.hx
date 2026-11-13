// `cond ? value : null` over an enum with a payload, an object or a
// structure yields the value itself, and null when the condition fails.
enum TernaryStep {
	Paths(first:Int);
	Meshes(count:Int, name:String);
}

class TernaryBox {
	public var v:Int;

	public function new(v:Int)
		this.v = v;
}

class TernaryReferenceBesideNull {
	static function last<T>(a:Array<T>):Null<T>
		return a.length > 0 ? a[a.length - 1] : null;

	static function main() {
		var out = [];
		var steps:Array<TernaryStep> = [Paths(1), Meshes(3, "a")];
		var none:Array<TernaryStep> = [];
		for (list in [steps, none]) {
			var s = list.length > 0 ? list[list.length - 1] : null;
			out.push(switch s {
				case Meshes(c, n): 'meshes $c $n';
				case Paths(f): 'paths $f';
				case null: "null";
			});
		}
		var objs = [new TernaryBox(7)];
		var o = objs.length > 0 ? objs[0] : null;
		out.push(o == null ? "null" : "box " + o.v);
		var p = objs.length > 1 ? objs[1] : null;
		out.push(p == null ? "null" : "box " + p.v);
		var anon = {k: 4};
		var q = steps.length > 0 ? anon : null;
		out.push("anon " + q.k);
		var e = Meshes(5, "z");
		var r = steps.length > 0 ? null : e;
		out.push(r == null ? "null" : Std.string(r));
		out.push(Std.string(last(steps)));
		var got = out.join(", ");
		if (got != "meshes 3 a, null, box 7, null, anon 4, null, Meshes(3,a)")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
