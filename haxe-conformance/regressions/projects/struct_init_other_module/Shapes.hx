enum Edge {
	Flat;
	Cut(w:Float, d:Float);
}

@:structInit
class Notch {
	public final a:Float;
	public final n:Int;
	public final top:Edge;
	public final bottom:Edge;

	public function new(a = 0.0, n = 5, ?top:Edge, ?bottom:Edge) {
		this.a = a;
		this.n = n;
		this.top = top == null ? Flat : top;
		this.bottom = bottom == null ? Flat : bottom;
	}

	public function describe():String
		return a + "/" + n + "/" + top + "/" + bottom;
}
