// A generic instance read through a Dynamic view hands its function field
// arguments that went through Dynamic; a class or interface parameter takes
// the instance they box. (The result's way back through the view is git-bug
// 86952ca.)
interface ErasedViewInterfaceArgumentsValue {
	public var ptr(default, null):Int;
}

class ErasedViewInterfaceArgumentsRadius implements ErasedViewInterfaceArgumentsValue {
	public var ptr(default, null):Int = 1;
	public final tl:Float;

	public function new(tl:Float)
		this.tl = tl;
}

class ErasedViewInterfaceArgumentsTween<T> {
	final lerp:(T, T, Float) -> Null<T>;
	var from:T;
	var to:T;

	public function new(start:T, lerp:(T, T, Float) -> Null<T>) {
		from = to = start;
		this.lerp = lerp;
	}

	public function probe(v:T):Null<T> {
		from = to;
		to = v;
		return lerp(from, to, 0.5);
	}
}

class ErasedViewInterfaceArguments {
	static var seen = "";

	static function lerpRadius(a:ErasedViewInterfaceArgumentsValue, b:ErasedViewInterfaceArgumentsValue,
			t:Float):Null<ErasedViewInterfaceArgumentsValue> {
		seen += a.ptr + "" + b.ptr;
		var x:ErasedViewInterfaceArgumentsRadius = cast a, y:ErasedViewInterfaceArgumentsRadius = cast b;
		return new ErasedViewInterfaceArgumentsRadius(x.tl + (y.tl - x.tl) * t);
	}

	static function lerpRadiusClass(a:ErasedViewInterfaceArgumentsRadius, b:ErasedViewInterfaceArgumentsRadius,
			t:Float):Null<ErasedViewInterfaceArgumentsRadius> {
		return new ErasedViewInterfaceArgumentsRadius(a.tl + (b.tl - a.tl) * t);
	}

	static function main() {
		var a:ErasedViewInterfaceArgumentsValue = new ErasedViewInterfaceArgumentsRadius(8);
		var b:ErasedViewInterfaceArgumentsValue = new ErasedViewInterfaceArgumentsRadius(28);
		var erased:ErasedViewInterfaceArgumentsTween<Dynamic> = cast new ErasedViewInterfaceArgumentsTween(a, lerpRadius);
		var r = erased.probe(b);
		var c:ErasedViewInterfaceArgumentsTween<Dynamic> = cast new ErasedViewInterfaceArgumentsTween(new ErasedViewInterfaceArgumentsRadius(2),
			lerpRadiusClass);
		var rc = c.probe(new ErasedViewInterfaceArgumentsRadius(6));
		var got = seen + "," + (r != null) + "," + (rc != null);
		if (got != "11,true,true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
