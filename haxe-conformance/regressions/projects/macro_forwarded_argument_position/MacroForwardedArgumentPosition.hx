import rx.Reactive;
import rx.Signal;

// The argument of a forwarded typing macro is typed as itself, once.
class MacroForwardedArgumentPosition {
	static function main() {
		var x = "ab";
		var boxes = [signal(x), signal([x, x].join("")), signal(x.toUpperCase()), signal([for (i in 0...3) x].join("-"))];
		var got = [for (b in boxes) b.kind + ":" + b.value].join(" ") + " " + rx.Box.made;
		if (got != "String:ab String:abab String:AB String:ab-ab-ab 4")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
