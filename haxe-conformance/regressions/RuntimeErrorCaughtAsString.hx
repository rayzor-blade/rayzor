// A runtime error caught as a String binds a real string: its message.
class RuntimeErrorCaughtAsString {
	static var f:Null<() -> Int> = null;

	static function main() {
		var got = [];
		try
			f()
		catch (e:String)
			got.push(e.length > 0 && e.charAt(0) != "" ? "string" : "empty");
		try
			f()
		catch (e:Dynamic)
			got.push(Std.string(e).length > 0 ? "dynamic" : "empty");
		try
			throw "abc"
		catch (e:String)
			got.push(e);
		var text = got.join(",");
		if (text != "string,dynamic,abc")
			throw text;
		Sys.println("CONFORMANCE_OK");
	}
}
