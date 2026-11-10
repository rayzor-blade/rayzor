package abs;

class Flags {
	public static inline var A:Int = 1;
	public static inline var B:Int = 2;
	public static var counter:Int = 3;

	public static function sum():Int
		return A + B + counter;
}
