class BareClassTokenDependency {
    static function main() {
        var cls = Xml;
        if (Type.getClassName(cls) != "Xml") throw "bare class token";
        if (Type.resolveClass("Xml") != cls) throw "resolved class token";
        Sys.println("CONFORMANCE_OK");
    }
}
