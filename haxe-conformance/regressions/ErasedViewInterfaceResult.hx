// An interface result read through a Dynamic view of a generic is the object
// it wraps, and through the typed view an interface value with its fields.
interface ErasedViewInterfaceResultIV {
	public var ptr(default, null):Int;
}

class ErasedViewInterfaceResultCR implements ErasedViewInterfaceResultIV {
	public var ptr(default, null):Int;

	public function new(p:Int)
		ptr = p;
}

class ErasedViewInterfaceResultTw<T> {
	final lerp:(T, T, Float) -> Null<T>;
	var from:T;
	var to:T;

	public function new(start:T, lerp) {
		from = to = start;
		this.lerp = lerp;
	}

	public function probe(v:T):String {
		from = to;
		to = v;
		var r = lerp(from, to, 0.5);
		return r == null ? "null" : "value " + Std.string(r);
	}

	public function get(v:T):T {
		from = to;
		to = v;
		return lerp(from, to, 0.5);
	}
}

class ErasedViewInterfaceResult {
	static function lerpR(a:ErasedViewInterfaceResultIV, b:ErasedViewInterfaceResultIV, t:Float):Null<ErasedViewInterfaceResultIV>
		return new ErasedViewInterfaceResultCR(Std.int(a.ptr + (b.ptr - a.ptr) * t));

	static function main() {
		var out = [];
		var erased:ErasedViewInterfaceResultTw<Dynamic> = cast new ErasedViewInterfaceResultTw<ErasedViewInterfaceResultIV>(new ErasedViewInterfaceResultCR(8),
			lerpR);
		out.push(erased.probe((new ErasedViewInterfaceResultCR(28) : ErasedViewInterfaceResultIV)));
		var r:Dynamic = erased.get((new ErasedViewInterfaceResultCR(48) : ErasedViewInterfaceResultIV));
		out.push(Std.string(r.ptr));
		var c:ErasedViewInterfaceResultCR = cast r;
		out.push(Std.string(c.ptr));
		out.push(Std.string(Std.isOfType(r, ErasedViewInterfaceResultCR)));
		var typed = new ErasedViewInterfaceResultTw<ErasedViewInterfaceResultIV>(new ErasedViewInterfaceResultCR(8), lerpR);
		out.push(Std.string(typed.get(new ErasedViewInterfaceResultCR(28)).ptr));
		var got = out.join(",");
		if (got != "value ErasedViewInterfaceResultCR,38,38,true,18")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
