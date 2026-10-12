import haxe.macro.Context;
import haxe.macro.Expr;
using haxe.macro.TypeTools;

private enum Kind<T> {
    IntKind(value:Int):Kind<Int>;
    FloatKind(value:Float):Kind<Float>;
    StringKind(value:String):Kind<String>;
}
private enum Wrapped<T> {
    Wrap<C>(kind:Kind<C>, value:C):Wrapped<C>;
}
private enum Reordered<A, B> {
    Make<X, Y>(left:X, right:Y):Reordered<Y, X>;
}
private enum Duo<A, B> {
    Values(left:A, right:B);
}
private enum Nested<T> {
    Nest<C>(value:C, values:Array<C>):Nested<Array<C>>;
}
class GadtConstructorParameters {
    static macro function same(value:Expr, expected:Expr):Expr {
        var actual = Context.typeof(value).toString();
        var wanted = Context.typeof(expected).toString();
        return macro $v{actual == wanted};
    }
    static function extract<T>(value:Wrapped<T>):T {
        return switch (value) { case Wrap(_, payload): payload; };
    }
    static function main() {
        var value = Wrap(FloatKind(2.5), 2.5);
        if (!same(value, (null:Wrapped<Float>))) throw "constructor type";
        var result = extract(value);
        if (!same(result, (null:Float))) throw "generic return type";
        if (result != 2.5) throw "payload value";
        if (extract(Wrap(IntKind(9), 9)) != 9) throw "integer payload value";
        if (extract(Wrap(StringKind("nine"), "nine")) != "nine") throw "string payload value";
        var reordered = Make(3, "three");
        if (!same(reordered, (null:Reordered<String, Int>))) throw "reordered result types";
        switch (reordered) {
            case Make(left, right):
                if (!same(left, (null:Int)) || !same(right, (null:String))) throw "reordered pattern types";
                if (left != 3 || right != "three") throw "reordered payload values";
        }
        var duo = Values(4, "four");
        if (!same(duo, (null:Duo<Int, String>))) throw "multiple enum parameters";
        switch (duo) {
            case Values(left, right):
                if (left != 4 || right != "four") throw "multiple payload values";
        }
        var nested = Nest(1.25, [1.25, 2.5]);
        if (!same(nested, (null:Nested<Array<Float>>))) throw "nested result type";
        switch (nested) {
            case Nest(first, values):
                if (!same(first, (null:Float)) || !same(values, (null:Array<Float>))) throw "nested pattern types";
                if (first != 1.25 || values.length != 2 || values[1] != 2.5) throw "nested payload values";
        }
        trace("CONFORMANCE_OK");
    }
}
