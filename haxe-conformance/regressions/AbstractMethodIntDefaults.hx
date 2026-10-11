private abstract class DefaultedBase {
    abstract public function value(number:Int = 20):Int;
}

private class DefaultedChild extends DefaultedBase {
    public function new() {}

    public function value(number:Int = 20):Int {
        return number;
    }
}

class AbstractMethodIntDefaults {
    static function main() {
        var child = new DefaultedChild();
        var base:DefaultedBase = child;
        if (child.value() != 20 || child.value(12) != 12) throw "concrete defaults";
        if (base.value(7) != 7) throw "abstract explicit argument";
        if (base.value() != 20 || base.value(null) != 20) throw "abstract defaults";
        Sys.println("CONFORMANCE_OK");
    }
}
