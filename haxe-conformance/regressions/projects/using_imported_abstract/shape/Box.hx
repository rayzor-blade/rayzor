package shape;

@:using(shape.Box)
abstract Box(Array<Int>) from Array<Int> to Array<Int> {
	public static function describe(b:Box):String {
		return "box" + (b : Array<Int>).length;
	}
}
