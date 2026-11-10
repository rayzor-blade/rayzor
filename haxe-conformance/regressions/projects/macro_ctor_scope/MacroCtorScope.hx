import lib.Ast;

@:build(lib.Mac.build())
class Holder {}

class MacroCtorScope {
	static function main() {
		var want = "has=true desc=M first=vertex kind=is-vertex,not-vertex";
		if (Holder.TEXT != want)
			throw Holder.TEXT + " != " + want;
		var funs:Array<Fun> = [{name: "f", kind: Fragment}];
		var runtime = [
			"has=" + lib.Comp.hasStage(funs),
			"desc=" + lib.Comp.describe(Main),
			"first=" + lib.Link.first(),
			"kind=" + lib.Link.kindIs(Kind.Vertex) + "," + lib.Link.kindIs(Kind.Main),
		].join(" ");
		if (runtime != want)
			throw runtime + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
