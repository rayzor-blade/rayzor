package utilities;

abstract ConstructedNumber(Int) to Int {
    public inline function new(value:Int) this = value * 3;
}
