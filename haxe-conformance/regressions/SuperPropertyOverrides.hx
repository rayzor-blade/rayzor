class SuperPropertyOverrides {
    static function main() {
        var child = new SuperPropertyChild();
        if (child.prop != 2) throw "super getter";
        if ((child.prop = 4) != 5) throw "super setter result";
        if (child.stored != 4) throw "super setter storage";
        if (child.callback(7) != "base7:child") throw "super callback";
        Sys.println("CONFORMANCE_OK");
    }
}
class SuperPropertyBase {
    public var stored:Int = 0;
    public var prop(get, set):Int;
    public var callback(get, never):Int->String;
    public function new() {}
    public function get_prop():Int return 1;
    public function set_prop(value:Int):Int return stored = value;
    public function get_callback():Int->String return (value:Int) -> "base" + value;
}
class SuperPropertyChild extends SuperPropertyBase {
    override public function get_prop():Int return super.prop + 1;
    override public function set_prop(value:Int):Int return (super.prop = value) + 1;
    override public function get_callback():Int->String {
        var inherited = super.callback;
        return (value:Int) -> inherited(value) + ":child";
    }
}
