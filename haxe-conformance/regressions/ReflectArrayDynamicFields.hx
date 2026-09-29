class ReflectArrayDynamicFields {
    static function main() {
        var object = {arr: [{foo: 1, bar: 42}]};
        var arr:Dynamic = Reflect.field(object, "arr");
        var item:Dynamic = arr[0];
        if (item.foo != 1) throw "dynamic anonymous field";
        if (item.bar != 42) throw "second dynamic anonymous field";
        if (Reflect.field(item, "foo") != item.foo) throw "Reflect.field mismatch";
        Sys.println("CONFORMANCE_OK");
    }
}
