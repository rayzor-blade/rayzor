private abstract Deferred<T>(() -> T) {
    inline function new(value:() -> T) this = value;
    @:from static function fromValue<T>(value:T):Deferred<T>
        return new Deferred(() -> value);
    @:to function get():T return this();
}

class AbstractLazyInitializers {
    static var text:Deferred<String> = "hello";
    static final number:Deferred<Float> = 1.5;
    var value:Deferred<Int> = 7;
    function new() {}
    static function main() {
        var a:String = text;
        var b:Float = number;
        var c:Int = new AbstractLazyInitializers().value;
        if (a != "hello" || b != 1.5 || c != 7) throw "lazy field conversion";
        Sys.println("CONFORMANCE_OK");
    }
}
