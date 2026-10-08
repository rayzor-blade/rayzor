using UsingMethodPrecedence.UsingFallbacks;

class UsingMethodPrecedence {
    static function main() {
        var root = new UsingRoot();
        if (root.one() != 1 || root.own() != 9) throw "declared method precedence";
        var child = new UsingChild();
        if (child.one() != -1) throw "super extension receiver";
        var instance = new UsingInterfaceInstance();
        if (!instance.checkInterface()) throw "implemented interface using";
        var typed:UsingChildInterface = instance;
        if (!typed.checkInterface()) throw "inherited interface using";
        var plain:UsingPlain<Array<Int>> = [0, 1];
        var specific:UsingSpecific<Array<Int>> = [0, 1];
        if (plain.count() != 2) throw "module extension";
        if (specific.count() != 0) throw "type extension precedence";
        if (specific.empty()) throw "module extension fallback";
        Sys.println("CONFORMANCE_OK");
    }
}

@:using(UsingMethodPrecedence.UsingExtensions)
class UsingRoot {
    public function new() {}
    public function own():Int return 9;
}

class UsingChild extends UsingRoot {
    public function new() super();
    public function one() return super.one() - 2;
}

@:using(UsingMethodPrecedence.UsingExtensions)
interface UsingRootInterface {}
interface UsingChildInterface extends UsingRootInterface {}
class UsingInterfaceInstance implements UsingChildInterface {
    public function new() {}
}

abstract UsingPlain<T>(T) from T to T {}
@:using(UsingMethodPrecedence.UsingExtensions)
abstract UsingSpecific<T>(T) from T to T {}

class UsingExtensions {
    public static function one(value:UsingRoot) return 1;
    public static function own(value:UsingRoot):Int return 1;
    public static function checkInterface(value:UsingRootInterface):Bool return true;
    public static function count<T>(value:Array<T>):Int return 0;
}

class UsingFallbacks {
    public static function count<T>(value:Array<T>):Int return value.length;
    public static function empty<T>(value:Array<T>):Bool return value.length == 0;
}
