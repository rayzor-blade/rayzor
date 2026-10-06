package helper;
abstract State(Int) {
    public static var count:Int = 3;
    public static var computed:Int = makeInitialValue();
    public static var label:String = "initial";
    public static var callback:Int->Int = function(value) return value + 2;
    public static final LIMIT:Int = 11;
    public static var value(get, set):Int;
    static function get_value():Int return count;
    static function set_value(next:Int):Int return count = next;
    static function makeInitialValue():Int return 8;
}
