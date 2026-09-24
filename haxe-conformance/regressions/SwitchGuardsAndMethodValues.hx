// Switch guards read the case's bindings; a no-argument constructor nested in
// a pattern is matched, not bound; closures in sibling cases capture `this`;
// a method value inside a closure keeps its receiver; a recursive method with
// an inferred String return maps as String; a Dynamic argument unifies with
// the type another argument gives a type parameter.
enum Quote { DQ; SQ; }
enum Lit { Str(s:String, ?q:Quote); Num(n:Int); Pair(a:Lit, items:Array<Int>); }
typedef Node = {kind:Kind};
enum Kind { Leaf(n:Int); List(items:Array<Node>); }

class Base {
	public function new() {}
	function same<X>(v:X, v2:X):Bool return v == v2;
}

class SwitchGuardsAndMethodValues extends Base {
	var pre = "<";

	function show(i:Int) return pre + i + ">";

	function lit(l:Lit):String
		return switch l {
			case Num(n) if (n > 0): "pos" + n;
			case Num(n): "num" + n;
			case Str(s, SQ): "single " + s;
			case Str(s, _): "other " + s;
			case Pair(Str(s, _), items): s + items.map(function(i) return show(i)).join(",");
			case Pair(_, items): "p" + items.map(function(i) return show(i)).join(",");
		}

	function node(n:Node)
		return switch n.kind {
			case Leaf(v): "<" + v + ">";
			case List(items): "(" + nodes(items) + ")";
		}

	function nodes(items:Array<Node>) return items.map(node).join(",");

	function opt<T>(v:T, f:T->String) return v == null ? "" : f(v);

	function viaClosure(xs:Array<Int>) return xs.map(function(x) return opt(x, show)).join(",");

	function untyped(e, s) return same(e + "", s);

	static var failed = false;

	static function check(label:String, got:Dynamic, want:Dynamic) {
		if (got != want) {
			failed = true;
			Sys.println("FAIL " + label + ": got " + got + " want " + want);
		}
	}

	static function main() {
		var t = new SwitchGuardsAndMethodValues();
		check("guard", t.lit(Num(3)) + " " + t.lit(Num(-1)), "pos3 num-1");
		check("nested nullary", t.lit(Str("a", DQ)) + "/" + t.lit(Str("b", SQ)) + "/" + t.lit(Str("c")), "other a/single b/other c");
		check("sibling closures", t.lit(Pair(Str("s"), [1])) + " " + t.lit(Pair(Num(0), [2])), "s<1> p<2>");
		check("recursive map", t.node({kind: List([{kind: Leaf(1)}, {kind: Leaf(2)}])}), "(<1>,<2>)");
		check("method value in closure", t.viaClosure([1, 2]), "<1>,<2>");
		check("dynamic unifies", t.untyped("xy", "xy"), true);
		if (!failed)
			Sys.println("CONFORMANCE_OK");
	}
}
