package conformance;
class ReflectionBase {
    public var value:Int;
    public function new(value:Int) { this.value = value; }
    public function read():Int { return value; }
}
class ReflectionMetadata extends ReflectionBase {
    public static var marker:Int = 9;
    public function new(value:Int) { super(value); }
    public function own():Int { return value + 1; }
    static function main() {
        var name = "conformance.ReflectionMetadata";
        if (Type.getClassName(ReflectionMetadata) != name) throw "qualified name";
        if (Type.resolveClass(name) != ReflectionMetadata) throw "resolveClass";
        var obj:ReflectionMetadata = Type.createInstance(ReflectionMetadata, [42]);
        if (obj == null || obj.value != 42 || obj.read() != 42) throw "constructor";
        if (Type.getClass(obj) != ReflectionMetadata) throw "getClass";
        var fields = Type.getInstanceFields(ReflectionMetadata);
        if (!fields.contains("value") || !fields.contains("read") || !fields.contains("own")) throw "instance fields";
        if (fields.contains("__type_id") || fields.contains("new")) throw "synthetic fields";
        var statics = Type.getClassFields(ReflectionMetadata);
        if (!statics.contains("marker") || !statics.contains("main")) throw "static fields";
        fields[0] = "changed";
        if (Type.getInstanceFields(ReflectionMetadata).contains("changed")) throw "field array alias";
        trace("CONFORMANCE_OK");
    }
}
