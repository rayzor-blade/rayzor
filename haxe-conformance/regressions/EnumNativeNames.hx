@:native("ExternalEnum")
enum LocalEnum {
    @:native("ExternalCase")
    Case;
}

class EnumNativeNames {
    static function main() {
        if (Type.getEnumName(LocalEnum) != "ExternalEnum")
            throw "enum reflection name";
        if (Type.enumConstructor(LocalEnum.Case) != "ExternalCase")
            throw "constructor reflection name";
        if (Type.getEnumConstructs(LocalEnum).join(",") != "ExternalCase")
            throw "constructor list name";
        Sys.println("CONFORMANCE_OK");
    }
}
