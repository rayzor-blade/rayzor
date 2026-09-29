package unit;

class InterfaceTypeReflection {
    static function main() {
        if (Type.getClassName(Contract) != "unit.Contract") throw "qualified interface name";
        if (Type.resolveClass("unit.Contract") != Contract) throw "qualified interface lookup";
        var token:Dynamic = Contract;
        if (Type.getClassName(token) != "unit.Contract") throw "Dynamic interface token";
        Sys.println("CONFORMANCE_OK");
    }
}
