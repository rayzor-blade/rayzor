package dbg;

import haxe.Int64;

class Trace {
	public static function begin(node:Null<Int64>, label:String):String {
		return label + ":" + describe(node) + Int64NullParamLoweredLater.tag();
	}

	public static function describe(node:Null<Int64>):String {
		if (node == null)
			return "?";
		return show(node);
	}

	static function show(id:Int64):String {
		return Int64.toStr(id);
	}
}
