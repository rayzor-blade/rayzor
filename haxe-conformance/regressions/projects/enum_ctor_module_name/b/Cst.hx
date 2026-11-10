package b;

// Never loaded: `Cst` in b.Gen is the constructor, not this module.
abstract IR<T>(Int) {}

class CstFns {
	static function __init__()
		throw "b.Cst loaded";
}
