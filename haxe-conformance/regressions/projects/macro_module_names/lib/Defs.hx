package lib;

enum Kind {
	Alpha;
	Beta(n:Int);
}

class Util {
	public static inline var LIMIT = 7;

	public static function twice(x:Int):Int
		return x * 2;
}

// Shares its name with haxe.macro.Type and the std Type class.
enum Type {
	TInt;
	TFun(variants:Array<Int>);
}
