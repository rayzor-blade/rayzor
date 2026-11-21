// A bare constructor passed where an imported enum is expected names that
// enum's constructor, though this module's enum abstract has a value of the
// same name; where the abstract is expected, it names the value.
import pk.Side;

enum abstract Layout(String) from String to String {
	var Inline = "inline";
	var Split = "split";
}

class ExpectedEnumOverEnumAbstractValue {
	static function pane(side:Side):String
		return switch side {
			case Original: "original";
			case Modified: "modified";
			case Inline: "inline-side";
		}

	static function main() {
		var layoutNow:Void->Layout = () -> Inline;
		var side:Side = Inline;
		var got = pane(Inline) + " " + layoutNow() + " " + pane(side) + " " + (side == Inline);
		if (got != "inline-side inline inline-side true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
