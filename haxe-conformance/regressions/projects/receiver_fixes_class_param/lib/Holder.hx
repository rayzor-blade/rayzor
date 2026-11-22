package lib;

import lib.Reactive;

/** Stores each reactive in a Table<Dynamic> from a generic method, as a node keeps its bindings. **/
class Holder {
	var stored:Null<Table<Dynamic>> = null;

	public function new() {}

	public function keep<T>(key:Int, value:Reactive<T>):Void {
		if (stored == null)
			stored = new Table();
		stored.set(key, value);
	}

	public function readAll():Array<String> {
		var out = [];
		if (stored == null)
			return out;
		for (i in 0...stored.length) {
			var value = stored.valueAt(i);
			out.push(Reader.read(value));
		}
		return out;
	}
}
