class InheritedTypeArguments {
    static function main() {
        var child = new ArgumentChild<Int>([7]);
        var value = child.value;
        if (!Std.isOfType(value, Array)) throw "inherited array field type";
        if (value.length != 1 || value[0] != 7) throw "inherited array field value";
        var methodValue = child.get();
        if (!Std.isOfType(methodValue, Array) || methodValue[0] != 7) throw "inherited method type";
        var renamed = new ArgumentRenamedChild<Int>([11]);
        var inherited = renamed.value;
        if (!Std.isOfType(inherited, Array) || inherited[0] != 11) throw "transitive parameter binding";
        var fixed = new ArgumentStringChild("fixed");
        var text = fixed.value;
        if (!Std.isOfType(text, String) || text.toUpperCase() != "FIXED") throw "fixed parent argument";
        var property = new ArgumentPropertyChild<Int>([9]);
        var result = property.value;
        if (!Std.isOfType(result, Array)) throw "inherited array property type";
        if (result.length != 1 || result[0] != 9) throw "inherited array property value";
        Sys.println("CONFORMANCE_OK");
    }
}
private class ArgumentParent<T> {
    public var value:T;
    public function new(value:T) this.value = value;
    public function get():T return value;
}
private class ArgumentChild<T> extends ArgumentParent<Array<T>> {
    public function new(value:Array<T>) super(value);
}
private class ArgumentRenamedChild<U> extends ArgumentChild<U> {
    public function new(value:Array<U>) super(value);
}
private class ArgumentStringChild extends ArgumentParent<String> {
    public function new(value:String) super(value);
}
private class ArgumentPropertyParent<T> {
    public var value(get, never):T;
    public function new() {}
    public function get_value():T throw "base getter";
}
private class ArgumentPropertyChild<T> extends ArgumentPropertyParent<Array<T>> {
    var stored:Array<T>;
    public function new(value:Array<T>) {
        stored = value;
        super();
    }
    override public function get_value():Array<T> return stored;
}
