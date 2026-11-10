package lib;

class Alloc {
	public var seed:Int;
	public var merged(get, default):Array<Int>;
	public var tagged(get, null):String;

	public function new(seed:Int)
		this.seed = seed;

	inline function get_merged() {
		if (merged == null)
			merged = [seed];
		return merged;
	}

	// Inside its own accessor, `this.tagged` is the storage.
	function get_tagged() {
		if (this.tagged == null)
			this.tagged = "t" + seed;
		return this.tagged;
	}
}

class Use {
	public static function sum(a:Alloc):Int {
		var n = 0;
		for (m in a.merged)
			n += m;
		return n;
	}

	public static function push(a:Alloc, v:Int):Int {
		a.merged.push(v);
		return a.merged.length;
	}
}
