// A structure's function-typed field is called through its value, even when
// a static extension or another class's method shares the name.
using StructFnField.StructFnExt;

typedef StructFnHost = {final text:Void->String; final caret:Void->Int;};

class StructFnExt {
	public static function text(h:StructFnHost):String
		return "extension";
}

class StructFnOther {
	public function new() {}

	public function text():String
		return "other";
}

class StructFnSlotA {
	public var caret:Int = 7;

	public function new() {}
}

class StructFnSlotB {
	public var pad = 0;
	public var caret = 9;

	public function new() {}
}

class StructFnUser {
	final host:StructFnHost;

	public function new(h)
		host = h;

	public function typed():String {
		var at = host.caret(), text = host.text();
		return text.substring(0, at);
	}
}

class StructFnField {
	static function main() {
		var h:StructFnHost = {text: () -> "hello", caret: () -> 3};
		var slots = new StructFnSlotA().caret + new StructFnSlotB().caret;
		var got = new StructFnUser(h).typed() + " " + h.text() + " " + new StructFnOther().text() + " " + slots;
		if (got != "hel hello other 16")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
