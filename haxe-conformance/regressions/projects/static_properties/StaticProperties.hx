import staticproperties.ImportedValues;

private class OtherProperties {
    public static var value(get, never):Int;
    static function get_value():Int { return 99; }
}
class StaticProperties {
    static var reads:Int = 0;
    static var writes:Int = 0;
    @:isVar static var value(get, set):Int;
    static function get_value():Int { reads++; return value; }
    static function set_value(next:Int):Int { writes++; return value = next; }
    static function main() {
        value = 3;
        writes = 0;
        reads = 0;
        if (value != 3 || reads != 1) throw "unqualified static getter";
        if (StaticProperties.value != 3 || reads != 2) throw "qualified static getter";
        value = 8;
        if (writes != 1 || value != 8) throw "static setter";
        value += 2;
        if (writes != 2 || value != 10) throw "compound static property";
        var previous = value++;
        if (previous != 10 || value != 11 || writes != 3) throw "postfix static property";
        if (OtherProperties.value != 99) throw "accessor owner";
        ImportedValues.initialize();
        ImportedValues.reads = 0;
        if (ImportedValues.amount != 17 || ImportedValues.reads != 1) throw "imported static getter";
        ImportedValues.amount = 21;
        ImportedValues.amount += 1;
        if (ImportedValues.amount != 22 || ImportedValues.writes != 3) throw "imported static setter";
        trace("CONFORMANCE_OK");
    }
}
