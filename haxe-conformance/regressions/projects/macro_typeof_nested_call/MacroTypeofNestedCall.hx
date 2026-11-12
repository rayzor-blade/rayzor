// Context.typeof of an expression holding a macro call types what the call
// expands to, as an argument or parsed from a string too.
import TypeofMac.typeName;
import TypeofMac.typeNameOf;
import TypeofMac.wrap;

class MacroTypeofNestedCall {
	static function local(s:String):Array<String> {
		return [s];
	}

	static function main() {
		var got = [
			typeName(local("a")),
			typeName(wrap("a")),
			typeName(local(wrap("a").join(","))),
			typeNameOf("wrap('a')")
		].join(",");
		if (got != "Array<String>,Array<String>,Array<String>,Array<String>")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
