// A Dynamic value passed to a class or enum parameter arrives as the
// object or enum value, whether it comes from a parameter, a comprehension
// variable or a closure argument.
class DynBox {
	public var x:Int;

	public function new(x:Int)
		this.x = x;
}

enum DynShape {
	Dot(b:DynBox);
	Line(b:DynBox);
	Area(b:DynBox);
}

class DynamicArgToClassAndEnumFormal {
	static function box(b:DynBox):String
		return "b" + b.x;

	static function shape(s:DynShape):String
		return switch (s) {
			case Dot(b): "D" + b.x;
			case Line(b): "L" + b.x;
			case Area(b): "A" + b.x;
		}

	static function viaBox(v:Dynamic):String
		return box(v);

	static function viaShape(v:Dynamic):String
		return shape(v);

	static function main() {
		var typed = new DynBox(1);
		var held:Dynamic = new DynBox(2);
		var boxes:Array<Dynamic> = [new DynBox(3), new DynBox(4)];
		var shapes:Array<Dynamic> = [Dot(new DynBox(5)), Line(new DynBox(6)), Area(new DynBox(7))];
		var closure = function(d:Dynamic) return box(d);
		var got = [
			viaBox(typed),
			viaBox(held),
			viaBox(boxes[0]),
			[for (d in boxes) box(d)].join(","),
			closure(boxes[1]),
			[for (s in shapes) shape(s)].join(","),
			viaShape(shapes[2]),
			viaShape(Line(new DynBox(8))),
		].join(" ");
		if (got != "b1 b2 b3 b3,b4 b4 D5,L6,A7 A7 L8")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
