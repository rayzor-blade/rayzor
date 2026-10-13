enum Choice { Value(number:Int); }
class EnumSwitchReturn {
    static function fail():Dynamic throw "bad";
    static function choose(name:String) {
        return switch(name) {
            case "one": Value(42);
            default: fail();
        }
    }
    static function main() {
        var result = choose("one");
        trace(result);
        switch(result) { case Value(number): if(number != 42) throw "payload"; }
        trace("CONFORMANCE_OK");
    }
}
