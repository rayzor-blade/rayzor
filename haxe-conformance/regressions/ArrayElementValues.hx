enum ArrayElementLeft {
    Same;
    Left;
}

enum ArrayElementRight {
    Right;
    Same;
}

class ArrayElementValues {
    static function main() {
        var values:Array<Any> = [12, false, "text", {}, 1.5];
        if (values.length != 5) throw "empty array element";
        var last:Float = values[4];
        if (last != 1.5) throw "following array element";
        var object:Dynamic = values[3];
        if (object == null) throw "empty object is null";
        Reflect.setField(object, "number", 17);
        if (Reflect.field(object, "number") != 17) throw "empty object field";
        var objects:Array<Dynamic> = [{}, {field:1}, {}];
        if (objects.length != 3 || objects[0] == null || objects[2] == null) {
            throw "object array positions";
        }
        var nested:Array<Array<Dynamic>> = [[{}], [{}]];
        if (nested.length != 2 || nested[0].length != 1 || nested[1][0] == null) {
            throw "nested object array";
        }
        var choices:Array<ArrayElementRight> = [Same];
        if (choices[0] != ArrayElementRight.Same) throw "array enum context";
        if (Type.getEnum(choices[0]) != ArrayElementRight) throw "array enum identity";
        Sys.println("CONFORMANCE_OK");
    }
}
