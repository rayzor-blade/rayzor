private enum HeldValue { Payload(text:String); }
private class ReturnBox<T> {
    var value:T;
    public function new(value:T) this.value = value;
    public function first():Null<T> return value;
    public function get():T return cast first();
}
class GenericDynamicReturnValues {
    static function field<T>(value:Null<T>):{value:T} return {value:value};
    static function main() {
        var value:Dynamic = Payload("held");
        var box = new ReturnBox<Dynamic>(value);
        var object:Dynamic = {};
        Reflect.setField(object, "value", box.get());
        if (!Type.enumEq(Reflect.field(object, "value"), value)) throw "dynamic enum return";
        var number:Dynamic = 1.5;
        var numbers = new ReturnBox<Dynamic>(number);
        if (numbers.get() != 1.5) throw "dynamic float return";
        if (field(number).value != 1.5) throw "dynamic float field";
        if (!Type.enumEq(field(value).value, value)) throw "dynamic enum field";
        Sys.println("CONFORMANCE_OK");
    }
}
