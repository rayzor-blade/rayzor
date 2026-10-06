private abstract BoxedNumber(NumberStorage) {
    inline private function new(value) {
        this = value;
    }
    inline public static function make(a:Int, b:Int):BoxedNumber {
        return new BoxedNumber(cast a + b);
    }
    @:op(A * B) inline public function multiply(other:BoxedNumber):BoxedNumber {
        return new BoxedNumber(cast this.get() * other.get().get());
    }
    public function get() return this;
}

private abstract NumberStorage(Dynamic) {
    public function get() return this;
}

class DynamicAbstractArithmetic {
    static function main() {
        var a = BoxedNumber.make(1, 2);
        var b = BoxedNumber.make(3, 4);
        if (a.get().get() != 3 || b.get().get() != 7) throw "abstract sums";
        var product = a * b;
        if (product.get().get() != 21) throw "abstract product";
        if (Std.string(product.get().get()) != "21") throw "abstract Dynamic storage";
        var again = product * a;
        if (again.get().get() != 63) throw "repeated abstract product";
        if (BoxedNumber.make(-6, 2).get().get() != -4) throw "negative abstract sum";
        if (Std.string(BoxedNumber.make(-6, 2).get().get()) != "-4") throw "negative abstract storage";
        Sys.println("CONFORMANCE_OK");
    }
}
