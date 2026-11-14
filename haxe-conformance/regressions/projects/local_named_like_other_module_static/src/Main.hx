class Box {
	public var value:Int;

	public function new(value:Int)
		this.value = value;

	public function get():Int
		return value;
}

// A local named like another module's static is the local, in a closure too.
class Main {
	static function main() {
		var revision = new Box(5);
		var f = () -> revision.get() + 1;
		var got = revision.get() + " " + f() + " " + p.Atlas.bump();
		if (got != "5 6 1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
