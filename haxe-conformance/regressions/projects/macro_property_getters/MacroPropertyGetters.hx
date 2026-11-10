import lib.Alloc;

@:build(lib.Mac.build())
class Holder {}

class MacroPropertyGetters {
	static function main() {
		var want = "sum=5 push=2 sum2=12 tag=t5";
		if (Holder.TEXT != want)
			throw Holder.TEXT + " != " + want;
		var a = new Alloc(5);
		var runtime = "sum=" + Use.sum(a) + " push=" + Use.push(a, 7) + " sum2=" + Use.sum(a) + " tag=" + a.tagged;
		if (runtime != want)
			throw runtime + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
