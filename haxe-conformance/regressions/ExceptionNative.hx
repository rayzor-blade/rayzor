// haxe.Exception.native is the native value it wraps, or the exception itself.
class ExceptionNative {
	static function main() {
		var e = new haxe.Exception("boom");
		var wrapped = new haxe.Exception("outer", null, "raw");
		var caught = "";
		try
			throw new haxe.Exception("t")
		catch (x:haxe.Exception)
			caught = x.message + " " + (x.native == x);
		var got = (e.native == e) + " " + wrapped.native + " " + caught;
		if (got != "true raw t true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
