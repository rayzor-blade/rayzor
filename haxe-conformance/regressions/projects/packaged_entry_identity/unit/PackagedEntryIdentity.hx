package unit;

class PackagedEntryIdentity {
    var comparer:Int->Int->Int = PackagedEntryIdentity.subtract;
    public function new() {}

    static function subtract(left:Int, right:Int):Int return left - right;

    static function check(label:String, value:Bool) {
        if (!value) throw label;
    }

    function test() {
        check("static function field", this.comparer(7, 2) == 5);
        check("imported Iterable alias", Lambda.count(Xml.parse("<!-- child -->")) == 1);
        check("packaged abstract accessor", Int64.MIN.present());
        check("qualified stdlib abstract", haxe.Int64.toStr(haxe.Int64.make(1, 2)) == "4294967298");
        var values:Map<String, EntryValue> = ["text" => "payload", "number" => 12];
        check("String conversion", values["text"].render() == "payload");
        check("Int conversion", values["number"].render() == "12");
        check("first static write", EntryCounter.next() == 1);
        check("second static write", EntryCounter.next() == 2);
        check("static read", EntryCounter.value == 2);
        var sibling = new EntrySibling(37);
        check("sibling class", sibling.read() == 37);
        check("imported class", OtherEntry.answer() == 52);
        Sys.println("CONFORMANCE_OK");
    }

    static function main() {
        new PackagedEntryIdentity().test();
    }
}

private class EntryCounter {
    public static var value = 0;
    public static function next():Int return ++value;
}

private class EntrySibling {
    var value:Int;
    public function new(value:Int) this.value = value;
    public function read():Int return value;
}

private abstract Int64(EntrySibling) {
    public function new(value:EntrySibling) this = value;
    public static var MIN(get, never):Int64;
    static function get_MIN():Int64 return new Int64(new EntrySibling(73));
    public inline function present():Bool return this != null;
}

private abstract EntryValue(EntryPayload) {
    public inline function new(value:EntryPayload) this = value;
    @:from public static function fromString(value:String):EntryValue return new EntryValue(Text(value));
    @:from public static function fromInt(value:Int) return new EntryValue(Number(value));
    public function render():String {
        return switch this {
            case Text(value): value;
            case Number(value): "" + value;
        };
    }
}

private enum EntryPayload {
    Text(value:String);
    Number(value:Int);
}
