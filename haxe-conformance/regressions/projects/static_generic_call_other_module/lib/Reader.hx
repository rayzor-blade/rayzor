package lib;

import lib.Value;
import lib.Writer;

// A static generic method that a class in another module calls through its
// class name, from a generic method of its own.
class Reader<T> {
	var writer:Writer;

	@:allow(lib.Writer)
	static function read<T>(value:Value<T>):T {
		return switch (value : Kind<T>) {
			case Const(v): v;
			case Bound(f): f();
		}
	}

	public static function initial(key:Int):Null<Dynamic>
		return key == 4 ? 7 : null;

	public static function describe(key:Int, v:Dynamic):String
		return key + "=" + Std.string(v);
}
