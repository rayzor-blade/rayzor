class ConstructorFieldInitialization {
    static function main() {
        var implicit = new InitializerImplicitChild();
        if (implicit.observed != 2 || implicit.child != 2) throw "implicit child initializer";
        var explicit = new InitializerExplicitChild();
        if (explicit.observed != 3 || explicit.child != 3) throw "explicit child initializer";
        var assigned = new InitializerAssignedChild();
        if (assigned.observed != 5 || assigned.child != 5) throw "pre-super assignment";
        var property = new InitializerPropertyChild(7);
        if (property.value != 7) throw "constructor virtual setter";
        Sys.println("CONFORMANCE_OK");
    }
}
private class InitializerParent {
    public var observed:Int;
    public function new() observe();
    function observe():Void observed = -1;
}
private class InitializerImplicitChild extends InitializerParent {
    public var child:Int = 2;
    override function observe():Void observed = child;
}
private class InitializerExplicitChild extends InitializerParent {
    public var child:Int = 3;
    public function new() super();
    override function observe():Void observed = child;
}
private class InitializerAssignedChild extends InitializerParent {
    public var child:Int = 4;
    public function new() {
        child = 5;
        super();
    }
    override function observe():Void observed = child;
}
private abstract class InitializerPropertyParent {
    public var value(get, set):Int;
    public function new(value:Int) this.value = value;
    abstract public function get_value():Int;
    abstract public function set_value(value:Int):Int;
}
private class InitializerPropertyChild extends InitializerPropertyParent {
    var values:Array<Int> = [];
    public function new(value:Int) super(value);
    public function get_value():Int return values[0];
    public function set_value(value:Int):Int {
        values.resize(0);
        values.push(value);
        return value;
    }
}
