package unit;

class EnumTypeReflection {
    static function main() {
        var types:Array<Dynamic> = [Choice];
        var token = types[0];
        if (token != Choice) throw "boxed enum token equality";
        if (Choice != token) throw "reversed enum token equality";
        if (Type.getEnumName(Choice) != "unit.Choice") throw "qualified enum name";
        if (Type.getEnumName(token) != "unit.Choice") throw "boxed enum name";
        if (Type.resolveEnum("unit.Choice") != Choice) throw "qualified enum resolution";
        Sys.println("CONFORMANCE_OK");
    }
}
