private interface Property {
    var value(get, set):Int;
}
private interface ExtendedProperty extends Property {
    function label():String;
}
private class OffsetProperty implements ExtendedProperty {
    public var padding:Int = 93;
    public var writes:Int = 0;
    var stored:Int = 7;
    public var value(get, set):Int;
    public function new() {}
    public function get_value():Int { return stored + 10; }
    public function set_value(value:Int):Int {
        writes++;
        stored = value - 10;
        return value;
    }
    public function label():String { return "offset"; }
}
private class DirectProperty implements Property {
    var stored:Int = 31;
    public var value(get, set):Int;
    public function new() {}
    public function get_value():Int { return stored; }
    public function set_value(value:Int):Int { return stored = value; }
}
private interface OrdinaryField {
    var value(default, default):Int;
}
private interface WriteOnly {
    var value(never, default):Int;
}
private class OrdinaryValue implements OrdinaryField implements WriteOnly {
    public var value:Int = 5;
    public function new() {}
    public function get_value():Int { return 999; }
}
class InterfaceProperties {
    static function read(property:Property):Int { return property.value; }
    static function write(property:Property, value:Int):Void { property.value = value; }
    static function writeOnly(property:WriteOnly, value:Int):Int {
        return { property.value = value; };
    }
    static function main() {
        var offset = new OffsetProperty();
        var direct = new DirectProperty();
        if (read(offset) != 17 || read(direct) != 31) throw "getter dispatch";
        write(offset, 23);
        write(direct, 42);
        if (read(offset) != 23 || read(direct) != 42) throw "setter dispatch";
        if (offset.writes != 1 || offset.padding != 93) throw "setter side effect";
        var extended:ExtendedProperty = offset;
        if (extended.value != 23 || extended.label() != "offset") throw "inherited getter";
        extended.value = 29;
        if (read(offset) != 29 || offset.writes != 2) throw "inherited setter";
        var plain = new OrdinaryValue();
        var ordinary:OrdinaryField = plain;
        if (ordinary.value != 5) throw "ordinary field read";
        ordinary.value = 8;
        if (ordinary.value != 8) throw "ordinary field write";
        if (writeOnly(plain, 11) != 11 || ordinary.value != 11) throw "write-only assignment value";
        trace("CONFORMANCE_OK");
    }
}
