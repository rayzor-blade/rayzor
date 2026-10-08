import utilities.ConstructorWrapper as Wrapper;
import utilities.ConstructedNumber;
import utilities.ConstructedObject;
import utilities.OwnConstructor;

class ImportedConstructorForwarding {
    static function main() {
        var number = new Wrapper<ConstructedNumber>(4);
        if (number.value() != 12) throw "imported abstract constructor";
        var instance = new utilities.ConstructorWrapper<ConstructedObject>(5);
        if (instance.value().number != 12 || ConstructedObject.calls != 1) throw "imported class constructor";
        var nested = new Wrapper<Wrapper<String>>("imported");
        if (nested.value().value() != "imported") throw "imported nested constructor";
        var own = new Wrapper<OwnConstructor>(3);
        if (own.value() != 10) throw "declared constructor precedence";
        Sys.println("CONFORMANCE_OK");
    }
}
