package helpers;
class Late {
    public static function reverse(value:Dynamic):String return Early.native(value);
    public static function text(value:String):String return value + Early.suffix();
    public static function number(value:Int):Int return value + 1;
    public static function decimal(value:Float):Float return value * 2;
    public static function enabled(value:Bool):Bool return value;
}
