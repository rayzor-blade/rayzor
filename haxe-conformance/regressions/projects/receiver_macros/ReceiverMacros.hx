import helpers.Receiver;
import helpers.MacroNumber;
using helpers.ReceiverExtensions;

class ReceiverMacros {
    static var calls = 0;
    static function make():Receiver {
        calls++;
        return new Receiver(3);
    }
    static function main() {
        if (make().identity().value != 3 || calls != 1) throw "receiver evaluation";
        var r = make();
        if (r.identity().typeName() != "helpers.Receiver") throw "receiver type";
        if (r.dupe() != "ordinary") throw "member priority";
        if ("ab".dupe() != "abab") throw "constant extension";
        if ((4 + 5).increment() != 10) throw "expression extension";
        if ((4 + 5).incrementArrow() != 11) throw "arrow extension";
        if ("foo".toUpperCase() != "FOO") throw "builtin member priority";
        var number = new MacroNumber<String>(7);
        if (number.typeName() != "helpers.MacroNumber<String>") throw "abstract receiver type";
        if (calls != 2) throw "compile time receiver evaluation";
        Sys.println("CONFORMANCE_OK");
    }
}
