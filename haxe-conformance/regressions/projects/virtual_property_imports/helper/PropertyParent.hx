package helper;
class PropertyParent {
    public var value(get, set):Float;
    public var label(get, never):String;
    public function new() {}
    public function get_value():Float return -1.0;
    public function set_value(value:Float):Float return -1.0;
    public function get_label():String return "parent";
}
