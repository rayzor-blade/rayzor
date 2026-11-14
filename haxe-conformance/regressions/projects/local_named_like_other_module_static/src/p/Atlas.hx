package p;

class Atlas {
	public static var revision(default, null) = 0;

	public static function bump():Int
		return ++revision;
}
