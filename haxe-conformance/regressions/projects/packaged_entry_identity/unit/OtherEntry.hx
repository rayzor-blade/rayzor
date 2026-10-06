package unit;

class OtherEntry extends EntryBase implements OtherContract {
    public function number():Int return 52;
    public static function answer():Int {
        var instance = new OtherEntry();
        if (!Std.isOfType(instance, OtherContract)) throw "interface startup";
        return instance.number();
    }
    public static function main() throw "wrong entry module";
}

private interface OtherContract {
    function number():Int;
}
