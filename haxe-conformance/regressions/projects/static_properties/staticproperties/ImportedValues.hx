package staticproperties;

class ImportedValues {
    public static var reads:Int = 0;
    public static var writes:Int = 0;
    @:isVar public static var amount(get, set):Int;
    static function get_amount():Int { reads++; return amount; }
    static function set_amount(next:Int):Int { writes++; return amount = next; }
    public static function initialize():Void { amount = 17; }
}
