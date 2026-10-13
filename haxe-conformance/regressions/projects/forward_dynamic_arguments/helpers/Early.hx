package helpers;
class Early {
    public static function text(value:Dynamic):String return Late.text(value);
    public static function number(value:Dynamic):Int return Late.number(value);
    public static function decimal(value:Dynamic):Float return Late.decimal(value);
    public static function enabled(value:Dynamic):Bool return Late.enabled(value);
    public static function native(value:String):String return value.toUpperCase();
    public static function suffix():String return ForwardDynamicArguments.suffix();
}
