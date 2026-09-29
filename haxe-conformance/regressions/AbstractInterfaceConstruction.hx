private interface Readable {
    var value(default, never):Int;
    function read():Int;
}
private typedef ReadableAlias = Readable;
private typedef Structural = { var value:Int; }
private class Reading implements Readable {
    public static var created:Int = 0;
    public var padding:Int = 93;
    public var value:Int;
    public function new(value:Int) { created++; this.value = value; }
    public function read():Int { return value + 1; }
}
private abstract View(Readable) {
    public var value(get, never):Int;
    function get_value():Int { return this.value; }
    public function new(source:Readable) { this = source; }
    public function readValue():Int { return this.value; }
    public function read():Int { return this.read(); }
}
private abstract AliasView(ReadableAlias) {
    public function new(source:ReadableAlias) { this = source; }
    public function read():Int { return this.read(); }
}
private abstract StructuralView(Structural) {
    public var value(get, set):Int;
    function get_value():Int { return this.value; }
    function set_value(value:Int):Int { return this.value = value; }
    public function new(source:Structural) { this = source; }
    public function readValue():Int { return this.value; }
    public function writeValue(value:Int):Void { this.value = value; }
}
class AbstractInterfaceConstruction {
    static function main() {
        var reading = new Reading(17);
        var view = new View(reading);
        if (view.readValue() != 17 || view.read() != 18) throw "class argument conversion";
        reading.value = 25;
        if (view.readValue() != 25 || view.read() != 26) throw "shared implementing object";
        var typed:Readable = reading;
        var alreadyTyped = new View(typed);
        if (alreadyTyped.read() != 26) throw "interface argument";
        var alias = new AliasView(new Reading(31));
        if (alias.read() != 32 || Reading.created != 2) throw "alias argument conversion";
        var structural = new StructuralView(new Reading(41));
        if (structural.readValue() != 41) throw "structural argument conversion";
        structural.writeValue(46);
        if (structural.readValue() != 46 || Reading.created != 3) throw "structural field write";
        if ((structural.value = 52) != 52 || structural.value != 52 || view.value != 25)
            throw "abstract accessor owner";
        trace("CONFORMANCE_OK");
    }
}
