abstract NumberSlot<T:Float>(T) from T to T {
    public inline function advance():Void this++;
}
abstract TaggedNumber<Tag, T:Float>(T) from T to T {
    public inline function advance():Void this++;
}
typedef DecimalNumber = NumberSlot<Float>;

class GenericAbstractScalars {
    static var stored:NumberSlot<Float> = 10.5;
    static var tagged:TaggedNumber<String, Float> = 20.25;
    static function check(value:Bool, label:String):Void { if (!value) throw label; }
    static function main():Void {
        stored.advance();
        check(stored == 11.5, "generic float static increment");
        var decimal:DecimalNumber = 2.5;
        decimal.advance();
        check(decimal == 3.5, "generic float local alias increment");
        var integer:NumberSlot<Int> = -3;
        integer.advance();
        check(integer == -2, "generic integer increment");
        tagged.advance();
        check(tagged == 21.25, "second generic parameter static storage");
        var second:TaggedNumber<Bool, Float> = 4.75;
        second.advance();
        check(second == 5.75, "second generic parameter local storage");
        Sys.println("CONFORMANCE_OK");
    }
}
