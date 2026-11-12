class PartB {
	public static function describe():String {
		var row:Row = {weight: 1.5, tags: ["x", "y"]};
		var p = new Patterns("b");
		return Patterns.NAME + " " + Patterns.items.length + " " + Patterns.RE.match("12px") + " "
			+ Patterns.twice(4) + " " + p.scaled() + " " + row.weight + row.tags.join("");
	}
}

private typedef Row = {weight:Float, tags:Array<String>};

private class Patterns {
	public static final NAME = "B";
	public static final items = ["one"];
	public static final RE = ~/[a-z]+/;

	public static function twice(v:Int):String
		return "twice" + v;

	var tag:String;

	public function new(tag:String)
		this.tag = tag;

	public function scaled():String
		return tag + tag;
}
