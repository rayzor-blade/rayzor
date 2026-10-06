package helper;
@:callable
abstract Checker(Bool->Void) {
    public static var calls:Int = 0;
    public static var check:Checker = new Checker();
    function new() {
        this = function(value:Bool):Void {
            calls++;
            if (!value) throw "check failed";
        };
    }
}
