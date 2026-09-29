private class MetaTypeClass { public function new() {} }
private enum MetaTypeEnum { First; Second; }

class MetaTypeValues {
    static function main() {
        var classValue:Dynamic = MetaTypeClass;
        var enumValue:Dynamic = MetaTypeEnum;
        var classType:Dynamic = Class;
        var enumType:Dynamic = Enum;
        var metaTypes:Array<Dynamic> = [Class, Enum];

        if (metaTypes.length != 2) throw "meta-type values omitted";
        if (!Std.isOfType(classValue, classType)) throw "class meta-type";
        if (!Std.isOfType(enumValue, enumType)) throw "enum meta-type";
        if (!Std.isOfType(classValue, metaTypes[0])) throw "array class meta-type";
        if (!Std.isOfType(enumValue, metaTypes[1])) throw "array enum meta-type";
        if (!Std.isOfType(MetaTypeClass, Class)) throw "direct class meta-type";
        if (!Std.isOfType(MetaTypeEnum, Enum)) throw "direct enum meta-type";
        if (Std.isOfType(classValue, classValue)) throw "class token as instance";
        if (Std.isOfType(enumValue, enumValue)) throw "enum token as value";
        if (Std.isOfType(MetaTypeClass, MetaTypeClass)) throw "direct class token as instance";
        if (Std.isOfType(MetaTypeEnum, MetaTypeEnum)) throw "direct enum token as value";
        Sys.println("CONFORMANCE_OK");
    }
}
