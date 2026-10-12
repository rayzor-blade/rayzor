import haxe.macro.Context;
using haxe.macro.TypeTools;

private enum abstract Kind<T>(String) {
    var Text:Kind<String>;
    var Integer:Kind<Int>;
}
private typedef Value<T> = { var kind:Kind<T>; var value:T; }
private enum Operation<A, B> {
    Add:Operation<Float, Float>;
    Equal:Operation<A, Bool>;
}

class GadtCaseRefinement {
    static macro function same(value:haxe.macro.Expr, expected:haxe.macro.Expr):haxe.macro.Expr {
        return macro $v{Context.typeof(value).toString() == Context.typeof(expected).toString()};
    }
    static function field<T>(item:Null<Value<T>>):T {
        var original = item.value;
        switch item.kind {
            case Text:
                if (!same(item.value, "")) throw "field String";
            case Integer:
                if (!same(item.value, 0)) throw "field Int";
        }
        if (!same(item.value, original)) throw "case refinement leaked";
        return item.value;
    }
    static function tuple<T>(item:Null<Value<T>>):T {
        return switch [item.kind, item.value] {
            case [Text, text]:
                if (!same(text, "")) throw "tuple String";
                text;
            case [Integer, integer]:
                if (!same(integer, 0)) throw "tuple Int";
                integer;
        };
    }
    static function object<T>(item:Null<Value<T>>):T {
        return switch item {
            case {kind: Text}:
                if (!same(item.value, "")) throw "object String";
                item.value;
            case {kind: Integer}:
                if (!same(item.value, 0)) throw "object Int";
                item.value;
        };
    }
    static function calculate<A, B>(operation:Operation<A, B>, left:A, right:A):B {
        return switch operation {
            case Add:
                if (!same(left, 0.0) || !same(right, 0.0)) throw "operand refinement";
                left + right;
            case Equal: left == right;
        };
    }
    static function main() {
        var text = {kind: Text, value: "payload"};
        var integer = {kind: Integer, value: 23};
        if (field(text) != "payload" || field(integer) != 23) throw "field values";
        if (tuple(text) != "payload" || tuple(integer) != 23) throw "tuple values";
        if (object(text) != "payload" || object(integer) != 23) throw "object values";
        if (calculate(Add, 12.5, 7.5) != 20) throw "addition";
        if (calculate(Equal, 12.5, 7.5)) throw "unequal Float";
        if (!calculate(Equal, "same", "same")) throw "equal String";
        if (calculate(Equal, "same", "different")) throw "unequal String";
        var first = new StringBuf();
        first.add("same");
        var second = new StringBuf();
        second.add("same");
        if (!calculate(Equal, first.toString(), second.toString())) throw "String content equality";
        trace("CONFORMANCE_OK");
    }
}
