private class Left {
    public var value(get, set):Int;
    var stored:Int = 0;
    public function new() {}
    function get_value():Int { return stored; }
    function set_value(v:Int):Int { return stored = v + 1; }
}
private class Right {
    public var value(get, set):Int;
    var padding:Int = 100;
    var stored:Int = 0;
    public function new() {}
    function get_value():Int { return stored; }
    function set_value(v:Int):Int { return stored = v + 2; }
}
private class Physical {
    public var value(get, null):Int;
    public function new() { value = 42; }
    function get_value():Int { return value; }
}
class Properties {
    static function main() {
        var a = new Left();
        var b = new Right();
        a.value = 10;
        b.value = 10;
        if (a.value != 11 || b.value != 12) throw "setter owner";
        var p = new Physical();
        if (p.value != 42) throw "physical property";
        trace("CONFORMANCE_OK");
    }
}
