private abstract Increasing(Int) to Int {
    public function new(value:Int) { this = value; step(); }
    inline function step():Void this++;
}
private abstract DebugText(String) to String {
    public inline function new(value:String) this = value;
}
@:multiType private abstract SelectedText(String) {
    public function new(value:String);
    public inline function tail(index:Int):String return this.substr(index);
    @:to public static inline function select(unused:String, value:String):DebugText return new DebugText(value);
}
class AbstractThisIncrement {
    static function main() {
        if ((new Increasing(2):Int) != 3) throw "constructor this increment";
        var text = new SelectedText("hello");
        if (text.tail(2) != "llo") throw "multitype constructor";
        Sys.println("CONFORMANCE_OK");
    }
}
