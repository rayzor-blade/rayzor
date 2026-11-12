class PartA {
	public static function describe():String {
		var row:Row = {label: "a", size: 3};
		var p = new Patterns(2);
		return Patterns.NAME + " " + Patterns.items.length + " " + Patterns.RE.match("12px") + " "
			+ Patterns.twice(4) + " " + p.scaled() + " " + row.label + row.size;
	}
}

private typedef Row = {label:String, size:Int};

private class Patterns {
	public static final NAME = "A";
	public static final items = [1, 2, 3];
	public static final RE = ~/^(\d+)px$/;

	public static function twice(v:Int):Int
		return v * 2;

	var factor:Int;

	public function new(factor:Int)
		this.factor = factor;

	public function scaled():Int
		return factor * 10;
}
