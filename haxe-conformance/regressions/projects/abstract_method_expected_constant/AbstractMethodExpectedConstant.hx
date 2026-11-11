// A bare enum abstract value passed to an abstract's method takes the parameter's abstract.
import Ids;
import Toks;

typedef Rec = {
	final a:Int;
}

@:forward
abstract Wrap(Rec) from Rec to Rec {
	public inline function get(token:Tok):String
		return (cast token : String);

	public function getPlain(token:Tok):String
		return (cast token : String);
}

class AbstractMethodExpectedConstant {
	static function idValue(i:Id):Int
		return (cast i : Int);

	static function main() {
		var w:Wrap = {a: 1};
		var got = [w.get(Background), w.getPlain(Background), w.get(Primary), "" + idValue(Background)].join(",");
		if (got != "background,background,primary,0")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
