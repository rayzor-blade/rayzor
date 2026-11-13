package b;

class MeshData {
	public static inline var STRIDE = 48;

	public static function stride():Int
		return STRIDE;

	public static function bytes(n:Int):Int
		return n * STRIDE;
}
