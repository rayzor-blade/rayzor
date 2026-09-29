package unit;

private class Hidden {
    public function new() {}
}

class PrivateClassReflection {
    static function main() {
        var value = new Hidden();
        if (Type.getClassName(Type.getClass(value)) != "unit._PrivateClassReflection.Hidden") {
            throw "private class RTTI name";
        }
        if (Type.getClassName(PrivateClassReflection) != "unit.PrivateClassReflection") {
            throw "public class RTTI name";
        }
        Sys.println("CONFORMANCE_OK");
    }
}
