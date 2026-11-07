private class Run {
    public final key:String;
    public var ctx:Int = 0;
    public var stopped(default, null) = false;
    public function new(key:String) this.key = key;
}
private class Decoy {
    public var pad:Int = 0;
    public var ctx:Int = 5;
    public var key:Int = 1;
}
class LambdaFindNullable {
    static function same<A>(x:A):Null<A> return x;
    static var running = new Map<Int, Array<Run>>();
    static function main() {
        running.set(1, [new Run("a")]);
        var list = running.get(1);
        var kept = Lambda.find(list, r -> r.key == "a" && !r.stopped);
        if (kept == null || kept.key != "a") throw "key through Null<Run>";
        kept.ctx = 6;
        if (kept.ctx != 6) throw "write through Null<Run>";
        var run = new Run("b");
        var again = same(run);
        if (again != run || again.key != "b") throw "Null<A> result of a generic function";
        var decoy = new Decoy();
        if (decoy.ctx + decoy.key != 6) throw "decoy";
        Sys.println("CONFORMANCE_OK");
    }
}
