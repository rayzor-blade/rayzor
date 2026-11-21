// A type check on an interface-typed value tests the object it wraps: another
// implementation of the interface is not the class, and null is nothing.
interface InterfaceValueTypeCheckIV {
	public var ptr(default, null):Int;
}

interface InterfaceValueTypeCheckOther {}

class InterfaceValueTypeCheckCR implements InterfaceValueTypeCheckIV {
	public var ptr(default, null):Int = 1;

	public function new() {}
}

class InterfaceValueTypeCheckSub extends InterfaceValueTypeCheckCR {}

class InterfaceValueTypeCheckAlt implements InterfaceValueTypeCheckIV implements InterfaceValueTypeCheckOther {
	public var ptr(default, null):Int = 2;

	public function new() {}
}

class InterfaceValueTypeCheck {
	static function isCR(a:InterfaceValueTypeCheckIV):Bool
		return Std.isOfType(a, InterfaceValueTypeCheckCR);

	static function isOther(a:InterfaceValueTypeCheckIV):Bool
		return Std.isOfType(a, InterfaceValueTypeCheckOther);

	static function main() {
		var cr:InterfaceValueTypeCheckIV = new InterfaceValueTypeCheckCR();
		var sub:InterfaceValueTypeCheckIV = new InterfaceValueTypeCheckSub();
		var alt:InterfaceValueTypeCheckIV = new InterfaceValueTypeCheckAlt();
		var none:InterfaceValueTypeCheckIV = null;
		var out = [
			isCR(cr), isCR(sub), isCR(alt), isCR(none),
			isOther(cr), isOther(alt),
			(alt is InterfaceValueTypeCheckAlt), (cr is InterfaceValueTypeCheckAlt),
		];
		var got = out.join(",");
		if (got != "true,true,false,false,false,true,true,false")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
