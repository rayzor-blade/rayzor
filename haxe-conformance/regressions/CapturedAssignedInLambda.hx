class Node {
	public var id:Int;

	public function new(id:Int)
		this.id = id;
}

class Box {
	public var node:Node;

	public function new(id:Int)
		node = new Node(id);
}

// An object a lambda makes and stores in a captured variable outlives the
// lambda: another lambda reads it later.
class CapturedAssignedInLambda {
	static function sequence(build:Void->Void, each:Int->Void) {
		build();
		for (f in 0...6)
			each(f);
	}

	static function main() {
		var card:Null<Box> = null;
		var build = () -> {
			card = new Box(42);
		};
		var seen = 0;
		sequence(build, frame -> {
			if (frame == 4)
				seen = card.node.id;
		});
		var later = new Box(7);
		if (seen != 42 || card.node.id != 42 || later.node.id != 7)
			throw seen + " " + card.node.id;
		Sys.println("CONFORMANCE_OK");
	}
}
