import lib.Light;

class A {
	public var view:Int;
	public var size:Int;

	public function new() {
		view = 1;
		size = 1;
	}
}

class B {
	public var view:Int;
	public var size:Int;

	public function new() {
		view = 2;
		size = 2;
	}
}

class InterfaceNullTypedef {
	static function run(f:Int->Void)
		f(3);

	static function main() {
		var lighting:Lighting = new Basic();
		var environment = lighting.environment();
		var typed:Null<Env> = lighting.environment();
		var out:Array<Int> = [environment.view, typed.view, lighting.shadows().size, new Basic().environment().view];
		run(encoder -> {
			var e = lighting.environment();
			if (e != null)
				out.push(e.view + e.levels);
			var s = lighting.shadows();
			if (s != null)
				out.push(s.size);
		});
		var got = out.join(",") + "," + (new A().view + new B().size);
		if (got != "7,7,64,7,9,64,3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
