private abstract Distance(Float) from Float to Float {
    public inline function new(value) this = value;
    public inline function replace(value:Float) this = value;
    public inline function increment(value:Float) this += value;
    public function get():Float return this;
    @:to public function toString():String return this + "m";
}

class AbstractFloatReceiver {
    static function main() {
        var value = new Distance(12.2);
        if (value.get() != 12.2) throw "float constructor";
        var text:String = value;
        if (text != "12.2m") throw "float conversion";
        value.replace(2.4);
        if (value.get() != 2.4) throw "float receiver assignment";
        value.increment(0.5);
        if (value.get() != 2.9) throw "float receiver compound assignment";
        Sys.println("CONFORMANCE_OK");
    }
}
