import lib.Renderer;

class Frame4 {
	final renderer:Renderer;
	public var out = 0;
	public var direct = 0;

	public function new(r) renderer = r;

	function suspend(f:Int->Void) f(10);

	public function run():Void {
		direct = @:privateAccess renderer.beginPass(1, 2);
		suspend(encoder -> {
			out = @:privateAccess renderer.beginPass(1, 2);
		});
	}

	public function run3(draw:Void->Void):Int {
		var view = renderer.maskTarget(6);
		var got = 0;
		suspend(encoder -> {
			got = @:privateAccess renderer.beginPass(encoder, view);
			@:privateAccess {
				var inner = renderer.beginPass(1, view);
				got += inner;
			}
			draw();
		});
		return got;
	}

	public function run2():Int {
		var local = 0;
		suspend(encoder -> {
			local = renderer.maskTarget(encoder);
		});
		return local;
	}
}

class LambdaPrivateAccess {
	static function main() {
		var f = new Frame4(new Renderer());
		f.run();
		var calls = 0;
		var got = f.direct + "," + f.out + "," + f.run2() + "," + f.run3(() -> calls++) + "," + calls;
		if (got != "3,3,10,23,1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
