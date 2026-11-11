package p;

class User {
	public static function make():Thing
		return new Thing();

	public static function describe():String {
		var t = new Thing();
		return t.tag + " " + t.n;
	}
}
