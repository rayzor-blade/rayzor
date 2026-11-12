package lib;

class Doc {
	static final byMarkup = new Map<String, Doc>();

	public final id:Int;

	function new(id:Int)
		this.id = id;

	public static function make(m:String):Doc {
		var k = byMarkup.get(m);
		if (k != null)
			return k;
		var d = new Doc(7);
		byMarkup.set(m, d);
		return d;
	}
}
