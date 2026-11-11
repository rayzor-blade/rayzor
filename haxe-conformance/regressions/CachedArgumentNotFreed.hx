// A call result passed straight to a function is not that call's to free: a cache may hold it.
class Color {
	public var rgb:Int;

	public function new(rgb:Int)
		this.rgb = rgb;
}

class Cache {
	static var colors = new Map<Int, Color>();

	public static function color(rgb:Int):Color {
		var c = colors.get(rgb);
		if (c == null) {
			c = new Color(rgb);
			colors.set(rgb, c);
		}
		return c;
	}
}

class Holder {
	public var held:Dynamic;

	public function new() {}

	public function style<T>(value:T):Void
		held = value;
}

class CachedArgumentNotFreed {
	static inline function write<T>(h:Holder, value:T):Void
		h.style(value);

	static function writeNI<T>(h:Holder, value:T):Void
		h.style(value);

	static function main() {
		var h = new Holder();
		var before = Cache.color(0xff0000);
		write(h, Cache.color(0xff0000));
		writeNI(h, Cache.color(0xff0000));
		h.style(Cache.color(0xff0000));
		var after = Cache.color(0xff0000);
		var got = (before == after) + " " + after.rgb + " " + (h.held : Color).rgb;
		if (got != "true 16711680 16711680")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
