package helpers;

import helpers.ConversionChecks.MacroChecks;
import helpers.ConversionChecks.MacroChecks as Checks;

class ModuleMacroPaths {
    static function require(value:Bool, label:String) {
        if (!value) throw label;
    }

    static function main() {
        require(ConversionChecks.MacroChecks.describe(12) == "Int", "module path");
        require(MacroChecks.describe("text") == "String", "imported secondary class");
        require(Checks.describe(true) == "Bool", "imported secondary alias");
        require(Checks.describe(Checks.describe(12)) == "String", "nested typing macro");
        require(!Checks.rejects(Checks.describe(12)), "nested valid expression");
        require(Checks.rejects(Checks.describe()), "nested invalid macro arity");
        require(OrdinaryChecks.describe(12) == "ordinary", "ordinary static method");
        trace("CONFORMANCE_OK");
    }
}

private class OrdinaryChecks {
    public static function describe(value:Int):String return "ordinary";
}
