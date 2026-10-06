private class Date {
    public var value:Int;
    public function new(value:Int) this.value = value;
}
private class EReg {
    public var value:Int;
    public function new(value:Int) this.value = value;
}
class DateConstructorIdentity {
    static function main() {
        var own = new Date(7);
        if (own.value != 7) throw "local Date constructor";
        var imported = new custom.Date(9);
        if (imported.value != 109) throw "imported Date constructor";
        var regex = new EReg(12);
        if (regex.value != 12) throw "local EReg constructor";
        Sys.println("CONFORMANCE_OK");
    }
}
