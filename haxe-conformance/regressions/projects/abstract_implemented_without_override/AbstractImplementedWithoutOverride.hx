// An abstract method of a class in another module, implemented without
// `override` below an abstract intermediate class, fills the slot the base
// dispatches on.
import ui.Component;

class Card extends Component<{title:String}> {
	function render():String
		return "card";
}

abstract class Cell extends Component<{span:Int}> {
	public var span = 1;
}

class Td extends Cell {
	function render():String
		return "td";
}

class AbstractImplementedWithoutOverride {
	static function main() {
		var got = [new Card().build(), new Td().build(), new Td().direct()].join(",");
		if (got != "card,td,td")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
