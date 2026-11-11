enum Shape {
	Circle(r:Int);
	Square;
}

class OwnExt {
	static public function describe(s:Shape):String
		return switch s {
			case Circle(r): "circle " + r;
			case Square: "square";
		}

	static public function twice(s:Shape, n:Int):String
		return describe(s) + " x" + n;
}
