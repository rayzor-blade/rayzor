package lib;

import lib.Ast;

class Comp {
	public static function hasStage(funs:Array<Fun>):Bool
		return Lambda.exists(funs, f -> f.kind == Vertex || f.kind == Fragment);

	public static function describe(k:Kind):String {
		return switch (k) {
			case Vertex: "V";
			case Fragment: "F";
			case Main: "M";
		}
	}
}
