// A subclass with field initializers and no constructor of its own takes its
// parent's constructor arguments.
class InheritedCtorFieldInitBase {
	public var v:Int;

	public function new(v:Int) {
		this.v = v;
	}
}

class InheritedCtorFieldInitSub extends InheritedCtorFieldInitBase {
	public var extra:Int = 5;
	public var list:Array<Int> = [1];
}

class InheritedCtorFieldInit {
	static function main() {
		var s = new InheritedCtorFieldInitSub(2);
		var got = s.v + "," + s.extra + "," + s.list.length;
		if (got != "2,5,1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
