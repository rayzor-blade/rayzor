package lib;

class Box {
	public var x:Int;

	public function new(x:Int)
		this.x = x;
}

enum Five {
	A5(b:Box);
	B5(b:Box);
	C5(b:Box);
	D5(b:Box);
	E5(b:Box);
}

class Shapes {
	public static function show(e:Five):String
		return switch (e) {
			case A5(b): "A" + b.x;
			case B5(b): "B" + b.x;
			case C5(b): "C" + b.x;
			case D5(b): "D" + b.x;
			case E5(b): "E" + b.x;
		}
}
