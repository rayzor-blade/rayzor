package unit;

private enum Choice { First; Second; }

class TypeFieldNames {
    static function main() {
        var names = Type.getInstanceFields(FieldSource);
        names.sort(Reflect.compare);
        if (names.join("|") != "read|value") throw "instance field names";
        if (!names.contains("read")) throw "instance field contains";

        var statics = Type.getClassFields(FieldSource);
        if (statics.length != 1 || statics.join("|") != "Marker")
            throw "static field names";
        if (Type.getEnumConstructs(Choice).join("|") != "First|Second")
            throw "enum constructor names";
        trace("CONFORMANCE_OK");
    }
}
