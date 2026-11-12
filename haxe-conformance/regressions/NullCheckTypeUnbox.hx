// `(v : T)` of a `Null<T>` primitive is the value it holds, not its box.
enum NullCheckTypeEdge {
	Scoop(width:Float, ?radius:Float);
}

class NullCheckTypeBox {
	public var r:Null<Float>;

	public function new(r)
		this.r = r;
}

class NullCheckTypeUnbox {
	static function radius(e:NullCheckTypeEdge):Float {
		return switch e {
			case Scoop(_, r): r == null ? 0.0 : (r : Float);
		}
	}

	static function main() {
		var box = new NullCheckTypeBox(6);
		var i:Null<Int> = 7;
		var b:Null<Bool> = true;
		var f:Null<Float> = 2.5;
		var encoded:Array<Float> = [1, (f : Float)];
		var got = radius(Scoop(1, 6)) + " " + radius(Scoop(1)) + " " + (box.r == null ? 0.0 : (box.r : Float)) + " "
			+ ((i : Int) + 1) + " " + ((b : Bool) ? "yes" : "no") + " " + encoded;
		if (got != "6 0 6 8 yes [1,2.5]")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
