@answer(42) @tag @nested({text: "hello", values: [1, 2]})
@:keep
private class Annotated {
    @fieldTag("field") public var value:Int;
    @staticTag(true) public static var marker:Int = 1;
    public function new() { value = 0; }
}
@kind("interface") private interface TaggedInterface {
    @methodTag(3) function call():Void;
}
class RuntimeMetadata {
    static function main() {
        var metadata = haxe.rtti.Meta.getType(Annotated);
        if (metadata.answer[0] != 42) throw "type metadata";
        if (!Reflect.hasField(metadata, "tag")) throw "marker metadata";
        if (Reflect.hasField(metadata, "keep")) throw "compiler metadata leaked";
        if (metadata.nested[0].text != "hello") throw "nested metadata";
        var fields = haxe.rtti.Meta.getFields(Annotated);
        if (fields.value.fieldTag[0] != "field") throw "field metadata";
        var statics = haxe.rtti.Meta.getStatics(Annotated);
        if (statics.marker.staticTag[0] != true) throw "static metadata";
        if (haxe.rtti.Meta.getType(TaggedInterface).kind[0] != "interface") throw "interface metadata";
        if (haxe.rtti.Meta.getFields(TaggedInterface).call.methodTag[0] != 3) throw "interface field metadata";
        trace("CONFORMANCE_OK");
    }
}
