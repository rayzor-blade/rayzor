package lib;

// A generic method whose type argument a caller in another module does not
// pass, so the call keeps the template and its erased slot takes a word.
enum Reactive<T> {
	Const(v:T);
	Bound(f:Void->T);
}
abstract Prop<T>(Int) from Int to Int {}
class Shadow {
	public final layers:Array<Array<Float>> = [];
	public function new(x:Float) layers.push([x, 1.0]);
}
class Batch {
	public final writes:Array<Void->Void> = [];
	public function new() {}
	public function set(key:Int, w:Void->Void) writes.push(w);
}
class Node {
	public static inline function write<T>(n:Node, prop:Prop<T>, value:T):Void
		n.style(prop, value);
	public var batch:Null<Batch> = new Batch();
	var immediate = false;
	var added:Array<Int> = [];
	public var seen = "";
	public function new() {}
	function field(prop:Int, cornerShape:Bool):Int return prop == 66 ? 9 : prop;
	function owns(key:Int):Bool return false;
	function add(key:Int):Bool { var f = added.indexOf(key) < 0; added.push(key); return f; }
	function apply<T>(prop:Prop<T>, r:Reactive<T>, fresh:Bool):Void {
		switch (r) {
			case Const(v):
				var d:Dynamic = v;
				seen += (prop : Int) + ":" + (Std.isOfType(d, Shadow) ? "S" + (cast d : Shadow).layers.length : Std.string(d)) + ":" + fresh + " ";
			case Bound(_):
		}
	}
	public function style<T>(prop:Prop<T>, value:T, cornerShape = false):Bool {
		var key = field(prop, cornerShape);
		if (owns(key))
			return false;
		var fresh = add(key);
		if (batch != null && !immediate) {
			batch.set(key, () -> apply(prop, Const(value), fresh));
			return true;
		}
		apply(prop, Const(value), fresh);
		return true;
	}
}
