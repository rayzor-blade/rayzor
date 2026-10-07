package helpers;

abstract NumericCursor(Int) from Int to Int {
    public inline function hasNext():Bool return this < 3;
    public inline function next():Int {
        var previous = this;
        this++;
        return previous;
    }
    public inline function advance(amount:Int):Int {
        var previous = this;
        this += amount;
        return previous;
    }
}
