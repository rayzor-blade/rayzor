// `Iterator<T>` is a structure; a class object passed as one answers its
// members with its methods.
class CountDown {
	var n:Int;

	public function new(n:Int)
		this.n = n;

	public function hasNext():Bool
		return n > 0;

	public function next():Int
		return n--;
}

class IteratorMethodOnClassObject {
	static final pending = new Map<String, Bool>();

	static function busy():Bool
		return pending.keys().hasNext();

	static function main() {
		var out = [Std.string(busy())];
		pending.set("a", true);
		out.push(Std.string(busy()));
		var it:Iterator<Int> = new CountDown(3);
		while (it.hasNext())
			out.push(Std.string(it.next()));
		var arr:Iterator<Int> = [7, 8].iterator();
		while (arr.hasNext())
			out.push(Std.string(arr.next()));
		var got = out.join(",");
		if (got != "false,true,3,2,1,7,8")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
