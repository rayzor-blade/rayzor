package other;

private enum Item {
	Word(text:String);
	Space;
}

class Flow {
	public static function first():Dynamic
		return Word("w");
}
