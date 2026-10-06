package helper;
class PropertyChild extends PropertyParent {
    var stored:Float = 2.5;
    public function new() super();
    override public function get_value():Float return stored;
    override public function set_value(value:Float):Float return stored = value;
    override public function get_label():String return "child";
    public function parentValue():Float return super.value;
    public function parentSet(value:Float):Float return super.value = value;
}
