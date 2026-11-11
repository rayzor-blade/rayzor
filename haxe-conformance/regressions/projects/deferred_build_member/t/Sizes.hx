package t;

typedef SizesData = {
	final small:Float;
	final large:Float;
}

@:forward
abstract Sizes(SizesData) from SizesData to SizesData {
	public inline function get(size:Size):Float {
		return size.of(this);
	}

	public static function defaults():Sizes
		return {small: 4, large: 16};
}
