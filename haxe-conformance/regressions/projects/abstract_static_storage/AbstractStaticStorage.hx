import helper.State;
import helper.Checker;
class AbstractStaticStorage {
    static function main() {
        if (State.count != 3) throw "initial count";
        State.count = 5;
        if (State.count != 5) throw "mutable count";
        if (State.computed != 8) throw "dynamic initializer";
        State.computed = 9;
        if (State.computed != 9) throw "mutable computed";
        if (State.label != "initial") throw "initial string";
        State.label = "changed";
        if (State.label != "changed") throw "mutable string";
        if (State.callback(4) != 6) throw "callable static";
        if (StaticHolder.callback(4) != 7) throw "class callable static";
        if (StaticHolder.calls != 1) throw "class callable side effect";
        Checker.check(true);
        var check = Checker.check;
        check(true);
        if (Checker.calls != 2) throw "callable abstract static";
        if (State.LIMIT != 11) throw "final constant";
        if (State.value != 5) throw "property read";
        State.value = 7;
        if (State.count != 7 || State.value != 7) throw "property write";
        if (LateState.count != 13) throw "forward static";
        LateState.count = 14;
        if (LateState.count != 14) throw "forward mutable";
        Sys.println("CONFORMANCE_OK");
    }
}
abstract LateState(Int) {
    public static var count:Int = 13;
}
private class StaticHolder {
    public static var calls:Int = 0;
    public static var callback:Int->Int = function(value) {
        calls++;
        return value + 3;
    };
}
