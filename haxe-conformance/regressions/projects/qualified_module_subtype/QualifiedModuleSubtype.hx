// `pack.Module.SubType.member` names a type a module declares beside others,
// as a `case` pattern and as an expression.
class QualifiedModuleSubtype {
	static function factor(a:Int):Float {
		return switch a {
			case lib.Shapes.Align.Center: 0.5;
			case lib.Shapes.Align.Right: 1.0;
			case _: 0.0;
		}
	}

	static function main() {
		var got = factor(1) + " " + factor(2) + " " + factor(0) + "|" + (lib.Shapes.Align.Center : Int) + " "
			+ (lib.Shapes.Dir.Column : Int);
		if (got != "0.5 1 0|1 1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
