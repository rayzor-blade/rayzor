// A type's @:using extends its values in modules other than its own.
import shape.Box;

class UsingImportedAbstract {
	static function main() {
		var b:Box = [1, 2];
		var cs:haxe.CallStack = [];
		var got = b.describe() + "," + cs.toString().length;
		if (got != "box2,0")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
