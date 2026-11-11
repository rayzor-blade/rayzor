// A function literal in an array literal takes its parameter types from the element type.
import a.Node;

class ArrayLambdaParams {
	static function apply(setters:Array<Node->Void>, node:Node)
		for (s in setters)
			s(node);

	static function main() {
		var node = new Node();
		var width = new a.Node.Prop<Int>(4);
		var label = new a.Node.Prop<String>(9);
		apply([n -> n.set(width, 5), n -> n.set(label, "x")], node);
		var typed:Array<Node->Void> = [n -> n.set(width, 6)];
		apply(typed, node);
		var got = node.log.join(",");
		if (got != "4=5,9=x,4=6")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
