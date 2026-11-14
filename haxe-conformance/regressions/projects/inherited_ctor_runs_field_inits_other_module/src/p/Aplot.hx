package p;

// Builds a Child; Child names Aplot back, so Aplot compiles first.
class Aplot {
	public static inline var NAME = "x";

	public static function make():Child
		return new Child({name: NAME});
}
