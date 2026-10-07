import helpers.GeneratedTypes.Chosen;

class ImportedGenericBuild {
    static function first(values:Chosen<Int>):Int return values[0];
    static function main() {
        var ints:Chosen<Int> = [3, 4];
        var strings:Chosen<String> = ["imported"];
        if (first(ints) != 3 || strings[0].toUpperCase() != "IMPORTED") throw "imported generic build";
        var constructed = new Chosen<Int>();
        constructed.push(5);
        if (constructed[0] != 5) throw "generic build constructor";
        Sys.println("CONFORMANCE_OK");
    }
}
