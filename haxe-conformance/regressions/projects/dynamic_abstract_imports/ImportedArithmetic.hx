import helpers.NumericWrapper;
class ImportedArithmetic {
    static function main() {
        var a = NumericWrapper.make(1, 2);
        var b = NumericWrapper.make(3, 4);
        var c = a * b;
        if (c.get().get() != 21) throw "imported abstract product";
        Sys.println("CONFORMANCE_OK");
    }
}
