// A structInit literal applies field initializers to omitted fields, and a
// declared constructor takes the literal's fields as arguments by name.
@:structInit
private class WithInit {
	public var x:Int = 100;
	public var y:Int;
	public final f:Float = 2;
}

@:structInit
private class WithCtor {
	public var x:Int;
	public var y:String;

	public function new(x:Int = 123, y:String = "hi") {
		this.x = x;
		this.y = y;
	}
}

class StructInitDefaults {
	static function main() {
		var a:WithInit = {y: 20};
		var b:WithCtor = {};
		var c:WithCtor = {y: "yo"};
		var got = a.x + "," + a.y + "," + a.f + "," + b.x + "," + b.y + "," + c.x + "," + c.y;
		trace(got == "100,20,2,123,hi,123,yo" ? "CONFORMANCE_OK" : "FAIL " + got);
	}
}
