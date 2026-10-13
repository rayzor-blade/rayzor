private class TokenFields {
    public static var label = "original";
}
private enum TokenChoice { First; Second; }
private enum TokenKey {
    ClassKey(type:Class<Dynamic>);
    EnumKey(type:Enum<Dynamic>);
}
class ClassTokenFields {
    static function read<T>(type:Class<T>, name:String):Dynamic {
        return Reflect.field(type, name);
    }
    static function write<T>(type:Class<T>, name:String, value:Dynamic):Void {
        Reflect.setField(type, name, value);
    }
    static function erase<T>(type:Class<T>):Dynamic return type;
    static function eraseEnum<T>(type:Enum<T>):Dynamic return type;
    static function main() {
        if (read(TokenFields, "label") != "original") throw "generic class field";
        write(TokenFields, "label", "updated");
        if (TokenFields.label != "updated") throw "generic class write";
        var erased = erase(TokenFields);
        if (Reflect.field(erased, "label") != "updated") throw "erased class field";
        var restored:Class<TokenFields> = erased;
        if (Type.getClassName(restored) != Type.getClassName(TokenFields)) throw "class identity";
        var enumType:Enum<TokenChoice> = eraseEnum(TokenChoice);
        if (Type.getEnumName(enumType) != Type.getEnumName(TokenChoice)) throw "enum identity";
        if (Reflect.field(123, "label") != null) throw "integer is not a class";
        var absent:Class<TokenFields> = null;
        if (erase(absent) != null) throw "null class token";
        if (Reflect.compare(erase(String), erase(TokenFields)) == 0) throw "distinct tokens";
        if (Reflect.compare(erase(TokenFields), erase(TokenFields)) != 0) throw "same token";
        if (Reflect.isEnumValue(eraseEnum(TokenChoice))) throw "enum type is not an enum value";
        if (!Reflect.isEnumValue(TokenChoice.First)) throw "enum value";
        if (!Reflect.isEnumValue(TokenChoice.Second)) throw "second enum value";
        var keys:Map<TokenKey, String> = new Map();
        keys.set(ClassKey(String), "string");
        keys.set(ClassKey(TokenFields), "class");
        keys.set(EnumKey(TokenChoice), "enum");
        if (keys.get(ClassKey(String)) != "string") throw "string type key";
        if (keys.get(ClassKey(TokenFields)) != "class") throw "class type key";
        if (keys.get(EnumKey(TokenChoice)) != "enum") throw "enum type key";
        trace("CONFORMANCE_OK");
    }
}
