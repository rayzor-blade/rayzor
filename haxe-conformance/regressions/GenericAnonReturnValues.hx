private enum ReturnedValue { Payload(text:String); }
class GenericAnonReturnValues {
    static function field<T>(value:Null<T>):{value:T} {
        return {value:value};
    }
    static function main() {
        if (field(1.5).value != 1.5) throw "float field";
        if (field(true).value != true) throw "boolean field";
        if (field("text").value != "text") throw "string field";
        if (field(7).value != 7) throw "integer field";
        if (Std.string(field(Payload("enum")).value) != "Payload(enum)") throw "enum field";
        Sys.println("CONFORMANCE_OK");
    }
}
