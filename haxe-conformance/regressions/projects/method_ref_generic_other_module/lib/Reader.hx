package lib;

import lib.Value;

class Reader {
	public static function read<T>(value:Value<T>):T {
		return switch (value : Kind<T>) {
			case Const(v): v;
			case Bound(f): f();
		}
	}

	public static function describe(key:Int, v:Dynamic):String
		return key + "=" + Std.string(v);
}
