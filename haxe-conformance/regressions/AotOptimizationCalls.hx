class AotParent {
    public function new() {}
    public function apply(value:Int):Int return value + 1;
    public var score(get, never):Int;
    function get_score():Int return 7;
}

class AotChild extends AotParent {
    override public function apply(value:Int):Int return value + 2;
}

class AotOptimizationCalls {
    @:neverInline static function helper(value:Int):Int {
        if (value < 0) throw "expected";
        return value * 3;
    }

    static function main() {
        var receiver:AotParent = new AotChild();
        var method = receiver.apply;
        var total = receiver.score;
        for (i in 0...4) total += helper(method(i));
        try { helper(-1); } catch (_:Dynamic) { total++; }
        if (total != 50) throw "incorrect AOT call result";
        Sys.println("CONFORMANCE_OK " + total);
    }
}
