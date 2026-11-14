class Atlas {
	public static inline var revision = 7;
}

class Box {
	public var value:Int;

	public function new(value:Int)
		this.value = value;

	public function get():Int
		return value;
}

// A local named like another class's inline constant is the local, in a closure too.
class LocalNamedLikeConstant {
	static function main() {
		var revision = new Box(5);
		var f = () -> revision.get() + 1;
		var count = 3;
		var g = () -> count * 2;
		var got = revision.get() + " " + f() + " " + g() + " " + Atlas.revision;
		if (got != "5 6 6 7")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
