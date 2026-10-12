private interface PropertySource {
    public var value(get, set):Int;
}
private class PropertyValue implements PropertySource {
    var stored:Int = 5;
    public var value(get, set):Int;
    public function new() {}
    function get_value():Int return stored;
    function set_value(value:Int):Int return stored = value;
}
class ReflectedGenericEnumValues {
    static function make<T>(value:T):haxe.ds.Option<T> return Some(value);
    static function main() {
        var source:PropertySource = new PropertyValue();
        if (Reflect.getProperty(source, "value") != 5) throw "interface getter";
        Reflect.setProperty(source, "value", 10);
        if (source.value != 10 || Reflect.getProperty(source, "value") != 10) throw "interface setter";
        var constructor:(Dynamic)->haxe.ds.Option<Dynamic> = Reflect.field(haxe.ds.Option, "Some");
        var a = constructor("hello");
        var b = constructor("hello");
        if (Std.string(a) != "Some(hello)") throw "reflected generic string";
        if (!Type.enumEq(a, b)) throw "reflected generic equality";
        if (Type.enumParameters(a)[0] != "hello") throw "reflected generic parameter";
        var called:haxe.ds.Option<Dynamic> = Reflect.callMethod(null, constructor, [1.5]);
        if (Std.string(called) != "Some(1.5)" || Type.enumParameters(called)[0] != 1.5) throw "reflected generic Float";
        var direct:Dynamic = make(2.75);
        if (Std.string(direct) != "Some(2.75)" || Type.enumParameters(direct)[0] != 2.75) throw "erased generic Float";
        var text:Dynamic = make("text");
        if (Std.string(text) != "Some(text)" || Type.enumParameters(text)[0] != "text") throw "erased generic String";
        Sys.println("CONFORMANCE_OK");
    }
}
