package utilities;

@:forward.new
abstract OwnConstructor(Int) to Int {
    public inline function new(value:Int) this = value + 7;
}
