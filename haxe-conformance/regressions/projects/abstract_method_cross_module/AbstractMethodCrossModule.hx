// An abstract method implemented in another module dispatches to the implementation.
import ui.Sub;
import ui.Card;

class AbstractMethodCrossModule {
	static function main() {
		var got = new Sub().make() + " " + new Card().show("x");
		if (got != "6 [card x]")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
