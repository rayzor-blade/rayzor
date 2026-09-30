enum MixedEnum {
    A;
    B;
    C(value:Int);
}

enum PlainEnum {
    X;
    Y;
}

class EnumReflectionValues {
    static function main() {
        var mixed = Type.allEnums(MixedEnum);
        if (mixed.join("#") != "A#B") throw "mixed enum names";
        if (Type.enumConstructor(mixed[0]) != "A" || Type.enumIndex(mixed[1]) != 1)
            throw "mixed enum values";
        if (!Type.enumEq(Type.createEnum(MixedEnum, "A"), MixedEnum.A))
            throw "createEnum representation";
        if (!Type.enumEq(Type.createEnumIndex(MixedEnum, 1), MixedEnum.B))
            throw "createEnumIndex representation";

        var plain = Type.allEnums(PlainEnum);
        if (plain.join("#") != "X#Y") throw "plain enum names";
        Sys.println("CONFORMANCE_OK");
    }
}
