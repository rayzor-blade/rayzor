package t;

private enum abstract Align(Int) {
	var Left = 0;
	var Center = 1;
}

class Flow {
	public static function left():Int
		return cast Align.Left;

	public static function center():Int
		return cast Align.Center;
}
