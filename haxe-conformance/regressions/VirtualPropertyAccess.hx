class VirtualPropertyAccess {
    static var selections:Int = 0;
    static function select(value:PropertyBase):PropertyBase {
        selections++;
        return value;
    }
    static function main() {
        var derived = new PropertyDerived();
        var base:PropertyBase = derived;
        if (base.get_value() != 9) throw "virtual method";
        if (base.value != 9) throw "virtual getter";
        if ((base.value = 12) != 12) throw "virtual setter result";
        if (derived.value != 12 || base.value != 12) throw "virtual setter storage";
        if (derived.baseValue() != -1) throw "super getter";
        if (derived.baseSet(21) != -1 || derived.value != 12) throw "super setter";
        select(base).value += 3;
        if (selections != 1 || derived.value != 15) throw "compound property evaluation";
        var previous = select(base).value++;
        if (selections != 2 || previous != 15 || derived.value != 16) throw "postfix property evaluation";
        var concrete = new PropertyConcrete();
        var abstractBase:PropertyAbstractBase = concrete;
        if (abstractBase.value != 8) throw "abstract getter";
        abstractBase.value = 10;
        if (concrete.value != 10) throw "abstract setter";
        Sys.println("CONFORMANCE_OK");
    }
}
private class PropertyBase {
    public var value(get, set):Int;
    public function new() {}
    public function get_value():Int return -1;
    public function set_value(value:Int):Int return -1;
}
private class PropertyDerived extends PropertyBase {
    var stored:Int = 9;
    public function new() super();
    override public function get_value():Int return stored;
    override public function set_value(value:Int):Int return stored = value;
    public function baseValue():Int return super.value;
    public function baseSet(value:Int):Int return super.value = value;
}
private abstract class PropertyAbstractBase {
    public var value(get, set):Int;
    public function new() {}
    abstract public function get_value():Int;
    abstract public function set_value(value:Int):Int;
}
private class PropertyConcrete extends PropertyAbstractBase {
    var stored:Int = 8;
    public function new() super();
    public function get_value():Int return stored;
    public function set_value(value:Int):Int return stored = value;
}
