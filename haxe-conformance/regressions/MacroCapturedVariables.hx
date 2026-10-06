import haxe.macro.Context;
import haxe.macro.Expr;
using haxe.macro.TypeTools;
class MacroCapturedVariables {
    static macro function collected() {
        var seen = [];
        function mapper(t:haxe.macro.Type):haxe.macro.Type {
            seen.push(t.toString());
            return Context.getType("String");
        }
        var t = Context.typeof(macro (null:{a:Bool,b:Int}));
        t.map(mapper);
        seen.sort(Reflect.compare); return macro $v{seen.join(",")};
    }
    static macro function counted() {
        var count = 0;
        function next() return ++count;
        var result = next() + "," + next() + "," + count;
        return macro $v{result};
    }
    static macro function escaped() {
        function makeCounter() {
            var count = 0;
            return function() return ++count;
        }
        var first = makeCounter();
        var second = makeCounter();
        var result = first() + "," + first() + "," + second() + "," + first();
        return macro $v{result};
    }
    static macro function shadowed() {
        var value = 1;
        var read = function() return value;
        var write = function(v:Int) value = v;
        var inner = 0;
        {
            var value = 9;
            write(4);
            inner = read() + value;
        }
        var result = inner + "," + value;
        return macro $v{result};
    }
    static macro function outerWrites() {
        var value = 1;
        var read = function() return value;
        value = 7;
        return macro $v{read()};
    }
    static function check(actual:String, expected:String) {
        if (actual != expected) throw actual + " != " + expected;
    }
    static function main() {
        check(collected(), "Bool,Int");
        check(counted(), "1,2,2");
        check(escaped(), "1,2,1,3");
        check(shadowed(), "13,4");
        if (outerWrites() != 7) throw "lost outer write";
        Sys.println("CONFORMANCE_OK");
    }
}
