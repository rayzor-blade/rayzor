package lib;

import lib.Ast;

// Shares constructor names with lib.Ast.Kind; private, so only this module sees it.
private enum Stage {
	Undefined;
	Vertex;
	Fragment;
}

class Link {
	static function stageName(s:Stage):String
		return s == Vertex ? "vertex" : "other";

	public static function first():String
		return stageName(Vertex);

	public static function kindIs(k:Kind):String
		return k == Vertex ? "is-vertex" : "not-vertex";
}
