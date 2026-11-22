package lib;

class Cell<T> {
	final f:Void->T;

	public function new(f:Void->T)
		this.f = f;

	public function get():T
		return f();
}

enum ReactiveKind<T> {
	Fixed(value:T);
	Live(cell:Cell<T>);
}

abstract Reactive<T>(ReactiveKind<T>) from ReactiveKind<T> to ReactiveKind<T> {
	@:from public static inline function fromValue<T>(value:T):Reactive<T>
		return Fixed(value);

	@:from public static inline function fromCell<T>(cell:Cell<T>):Reactive<T>
		return Live(cell);
}

class Table<T> {
	final keys:Array<Int> = [];
	final values:Array<T> = [];

	public function new() {}

	public function set(key:Int, value:T):Void {
		keys.push(key);
		values.push(value);
	}

	public var length(get, never):Int;

	inline function get_length():Int
		return keys.length;

	public inline function valueAt(i:Int):T
		return values[i];
}

class Reader {
	public static function read(value:Reactive<Dynamic>):String
		return switch (value : ReactiveKind<Dynamic>) {
			case Fixed(v): "fixed " + Std.string(v);
			case Live(c): "live " + Std.string(c.get());
		};
}
