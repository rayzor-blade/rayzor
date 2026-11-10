@:build(y.Mac.build())
class Holder {}

class MacroTypedVarCtor {
	static function main() {
		var want = "y.Def:block2,y.Def:fn foo2,y.Def:fn x1,y.Def:block2,y.Def:block1,y.Def:break,y.Def:block3";
		if (Holder.TEXT != want)
			throw Holder.TEXT + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
