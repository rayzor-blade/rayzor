package lib;

import lib.Value;

class Writer {
	public var log:Array<String> = [];
	public var on = true;

	public function new() {}

	public function apply<T>(key:Int, value:Value<T>):Void {
		if (on)
			snap(key, null, value);
	}

	public function unset(key:Int):Void {
		var initial = lib.Reader.initial(key);
		if (initial != null)
			snap(key, 1, Const(initial));
	}

	function snap<T>(key:Int, from:Null<Dynamic>, to:Value<T>):Void {
		if (!on)
			return;
		log.push(lib.Reader.describe(key, from) + ">" + lib.Reader.describe(key, lib.Reader.read(cast to)));
	}
}
