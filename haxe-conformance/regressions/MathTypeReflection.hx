class MathTypeReflection {
    static function main() {
        if (Type.getClassName(Math) != "Math") throw "Math class name";
        if (Type.resolveClass("Math") != Math) throw "Math class resolution";
        var fields = Type.getClassFields(Math);
        if (fields.indexOf("PI") < 0 || fields.indexOf("abs") < 0) {
            throw "Math static fields";
        }
        Sys.println("CONFORMANCE_OK");
    }
}
