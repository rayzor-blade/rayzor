package lib;

// A generic class keeping closures in fields declared at its type parameter,
// drained through a Dynamic-typed list, as a reactive watch does.
class Ctx {
	public static var current = new Ctx();

	public function new() {}

	public static function capture<T>(fn:Void->T):Void->T {
		var context = current;
		return () -> context.run(fn);
	}

	public function run<T>(fn:Void->T):T {
		try {
			var result = fn();
			return result;
		} catch (e:haxe.Exception) {
			throw e;
		}
	}
}

class Watch<T> {
	static var changed:Array<Watch<Dynamic>> = [];

	final read:Void->T;
	final react:T->Void;
	final same:(T, T) -> Bool;
	var latest:T;
	var reacted:T;

	public function new(read:Void->T, react:T->Void, ?same:(T, T) -> Bool) {
		this.read = Ctx.capture(read);
		var context = Ctx.current;
		this.react = value -> context.run(() -> react(value));
		this.same = same != null ? same : (a, b) -> a == b;
		latest = this.read();
		reacted = latest;
		this.react(latest);
		changed.push(this);
	}

	public static function flush():Void {
		for (watch in changed) {
			watch.latest = watch.read();
			if (watch.same(watch.reacted, watch.latest))
				continue;
			watch.reacted = watch.latest;
			watch.react(watch.latest);
		}
	}
}
