class Handle {
	public final id:Int;

	public function new(id:Int)
		this.id = id;
}

typedef HandleAlias = Handle;

interface INamed {
	function name():String;
}

class Named implements INamed {
	final n:String;

	public function new(n:String)
		this.n = n;

	public function name():String
		return n;
}

interface IValue {
	var ptr(default, null):Handle;
	var alias(default, null):HandleAlias;
	var ids(default, null):Array<Int>;
	var named(default, null):INamed;
	var meta(default, null):{size:Int};
	var shift(default, null):Int->Int;
	var packed(default, null):Handle;
}

class Color implements IValue {
	public var ptr(default, null):Handle;
	public var alias(default, null):HandleAlias;
	public var ids(default, null):Array<Int>;
	public var named(default, null):INamed;
	public var meta(default, null):{size:Int};
	public var shift(default, null):Int->Int;
	// A native handle that is a packed value, not a pointer.
	public var packed(default, null):Handle;

	public function new(hex:Int) {
		ptr = new Handle(hex);
		alias = new Handle(hex + 1);
		ids = [hex];
		named = new Named("blue");
		meta = {size: hex - 5};
		shift = x -> x + hex;
		packed = cast 6;
	}
}

class Holder<T:IValue> {
	public var held:Handle;

	public function new(v:T) {
		held = take(v == null ? null : v.ptr);
	}

	public static function idOf<T:IValue>(v:T):Int
		return take(v.ptr).id;

	public static function aliasOf<T:IValue>(v:T):Int {
		var p = v.alias;
		return take(p).id;
	}

	public static function firstId<T:IValue>(v:T):Int
		return v.ids[0];

	public static function nameOf<T:IValue>(v:T):String
		return v.named.name();

	public static function sizeOf<T:IValue>(v:T):Int
		return v.meta.size;

	public static function callShift<T:IValue>(v:T):Int
		return v.shift(3);

	public static function readShift<T:IValue>(v:T):Int {
		var f = v.shift;
		return f(4);
	}

	public static function packedOf<T:IValue>(v:T):Int {
		var h:Handle = v.packed;
		return cast h;
	}

	static function take(h:Handle):Handle
		return h;
}

class InterfaceBoundedFieldReads {
	static function check(label:String, got:Int, want:Int):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		var blue = new Color(255);
		var h = new Holder<Color>(blue);
		check("class field passed on", h.held == blue.ptr ? 1 : 0, 1);
		check("class field read", h.held.id, 255);
		check("class field into a call", Holder.idOf(blue), 255);
		check("aliased field into a call", Holder.aliasOf(blue), 256);
		check("array field", Holder.firstId(blue), 255);
		check("interface field", Holder.nameOf(blue) == "blue" ? 1 : 0, 1);
		check("anonymous field", Holder.sizeOf(blue), 250);
		check("function field called", Holder.callShift(blue), 258);
		check("function field read", Holder.readShift(blue), 259);
		check("packed handle field", Holder.packedOf(blue), 6);
		Sys.println("CONFORMANCE_OK");
	}
}
