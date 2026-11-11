package w;

using w.Ast;

class Use {
	static function mk(n:String, ?q:Array<Int>):V
		return {id: 1, name: n, q: q};

	static function total(b:Built):Int
		return b.total;

	static var seen = new Map<V, Int>();

	static function visit(v:V):Bool {
		var known = seen.exists(v);
		seen.set(v, 1);
		v.name += "!";
		return known;
	}

	static function fault():String {
		try {
			var m:Map<Int, Int> = null;
			m.set(1, 2);
		} catch (e:Error) {
			return "user:" + e.msg;
		} catch (e:Dynamic) {
			return "other";
		}
		return "none";
	}

	public static function run():String {
		var a:V = {id: 1, name: "a", q: [3, 4]};
		var b = mk("b");
		var c = mk("c", [4]);
		var h:Holder = {v: {id: 2, name: "h", q: [5]}};
		var k:V = null;
		k = {id: 9, name: "k"};
		var first = visit(k);
		var second = visit(k);
		var parts:Array<Dynamic> = [
			a.has(3), a.has(9), b.has(3), c.has(4), Tools.has(a, 4), h.v.has(5),
			h.tag, Type.getClassName(Type.getClass(a)), Type.getClassName(Type.getClass(h.v)),
			total({a: 2, b: 5}), first, second, k.name, fault(),
		];
		return parts.join(",");
	}
}
