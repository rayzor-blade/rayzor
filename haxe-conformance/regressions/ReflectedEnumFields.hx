private enum FieldEnum { Payload(value:Int); }
private typedef CycleRecord = {link:Null<CycleEnum>};
private enum CycleEnum { Node(record:CycleRecord); }
class ReflectedEnumFields {
    static function main() {
        var value = Payload(42);
        var object:{value:Dynamic} = {value:null};
        Reflect.setField(object, "value", value);
        var typed:{value:FieldEnum} = cast object;
        if (typed.value != value) throw "typed enum field identity";
        var nullable:{value:Null<FieldEnum>} = cast object;
        if (nullable.value != value) throw "nullable enum field identity";
        var direct:{value:Null<FieldEnum>} = {value:value};
        if (direct.value != value) throw "raw enum field identity";
        var inferred = {value:null};
        inferred.value = value;
        if (!Type.enumEq(Reflect.field(inferred, "value"), value)) throw "inferred enum field identity";
        switch (Type.typeof(Reflect.field(inferred, "value"))) {
            case TEnum(_):
            default: throw "inferred enum field type";
        }
        var record = {link:null};
        var node = Node(record);
        record.link = node;
        if (!Type.enumEq(Reflect.field(record, "link"), node)) throw "cyclic enum field identity";
        switch (Type.typeof(Reflect.field(record, "link"))) {
            case TEnum(_):
            default: throw "cyclic enum field type";
        }
        Sys.println("CONFORMANCE_OK");
    }
}
