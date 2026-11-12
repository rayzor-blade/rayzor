package gen;

class Maker {
	public static function count():Int {
		var list = new gen.ListBox([1, 2, 3]);
		return list.value.length;
	}
}
