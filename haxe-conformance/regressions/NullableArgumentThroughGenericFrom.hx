// A conditional of type Null<Single> passed for `IntoValue<T>` where another
// argument binds T to Single converts through the generic @:from at Single.
#if rayzor
typedef NullableFromNumber = Single;
#else
typedef NullableFromNumber = Float;
#end

enum NullableFromSource<T> {
	Constant(value:T);
}

abstract NullableFromInto<T>(NullableFromSource<T>) from NullableFromSource<T> to NullableFromSource<T> {
	@:from public static inline function fromConst<T>(val:T):NullableFromInto<T>
		return Constant(val);
}

abstract NullableFromKey<T>(Int) {
	public static inline var Width:NullableFromKey<NullableFromNumber> = cast 1;
}

typedef NullableFromAttr = {?width:NullableFromNumber, ?height:NullableFromNumber};

class NullableArgumentThroughGenericFrom {
	static var stored = "";

	static function set<T>(key:NullableFromKey<T>, value:NullableFromInto<T>):Void
		switch (value : NullableFromSource<T>) {
			case Constant(v): stored = Std.string(v);
		}

	static function apply(fallback:Int, ?attr:NullableFromAttr):String {
		set(NullableFromKey.Width, attr != null && attr.width != null ? attr.width : (fallback : NullableFromNumber));
		return stored;
	}

	static function main() {
		var got = [apply(24), apply(24, {}), apply(24, {width: 50})].join(",");
		if (got != "24,24,50")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
