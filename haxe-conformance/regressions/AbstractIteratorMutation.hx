private abstract Counter(Int) from Int to Int {
    public inline function hasNext():Bool return this < 3;
    public inline function next():Int {
        var previous = this;
        this = this + 1;
        return previous;
    }
    public inline function advance(amount:Int):Int {
        var previous = this;
        this += amount;
        return previous;
    }
    public inline function twice(amount:Int):Void {
        this += amount;
        this += amount;
    }
    public inline function mask(bits:Int):Void this |= bits;
}
private abstract Fraction(Float) from Float to Float {
    public inline function step():Float {
        var previous = this;
        this += 0.5;
        return previous;
    }
}
private abstract RecordCursor({current:Int, last:Int}) {
    public var length(get, never):Int;
    inline function get_length():Int return this.last - this.current;
    @:arrayAccess public function get(i:Int):Int return this.current - i;
    public inline function new(start:Int, end:Int) this = {current:start, last:end};
    public inline function hasNext():Bool return this.current < this.last;
    public inline function next():Int return this.current++;
}
private class CounterHolder {
    public var value:Counter = 10;
    public function new() {}
}
class AbstractIteratorMutation {
    static var calls:Int = 0;
    static var holder:CounterHolder;
    static var values:Array<Counter>;
    static function getHolder():CounterHolder { calls++; return holder; }
    static function getValues():Array<Counter> { calls++; return values; }
    static function amount():Int { calls++; return 2; }
    static function check(value:Bool, label:String):Void { if (!value) throw label; }
    static function main():Void {
        var counter:Counter = 4;
        check(counter.advance(3) == 4 && counter == 7, "local writeback and result");
        check(counter.advance(2) == 7 && counter == 9, "fresh inline locals");
        counter.twice(amount());
        check(counter == 13 && calls == 1, "argument evaluated once");
        counter.mask(2);
        check(counter == 15, "void compound writeback");
        holder = new CounterHolder();
        check(getHolder().value.advance(3) == 10 && holder.value == 13 && calls == 3, "inline field receiver sequence");
        values = [20, 30];
        var index = 0;
        check(getValues()[index++].advance(4) == 20 && values[0] == 20 && values[1] == 34 && index == 2 && calls == 5, "inline array receiver sequence");
        var fraction:Fraction = 1.25;
        check(fraction.step() == 1.25 && fraction == 1.75, "float writeback");
        var text = "";
        for (value in (0:Counter)) text += value;
        check(text == "012", "value abstract iterator");
        var result = [for (value in new RecordCursor(0, 3)) value];
        check(result.length == 3 && result[0] == 0 && result[1] == 1 && result[2] == 2, "own protocol before array access");
        var sum = 0;
        for (value in (0:Counter)) {
            if (value == 0) continue;
            sum += value;
            if (value == 1) break;
        }
        check(sum == 1, "abstract iteration control flow");
        Sys.println("CONFORMANCE_OK");
    }
}
