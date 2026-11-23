package lib;

import lib.Value;

// A generic method that a module other than its own takes as a function value.
class Writer {
	public var log:Array<String> = [];

	public function new() {}

	public function apply<T>(key:Int, value:Value<T>):Void
		snap(key, value);

	public function put<T>(key:Int, value:T):Void
		log.push(key + ":" + Std.string(value));

	function snap<T>(key:Int, to:Value<T>):Void
		log.push(Reader.describe(key, Reader.read(cast to)));
}
