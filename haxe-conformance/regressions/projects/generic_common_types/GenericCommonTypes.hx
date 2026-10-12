import haxe.macro.Context;
import haxe.macro.Expr;
using haxe.macro.TypeTools;

private class CommonBase<T> {
    public var value:T;
    public function new(value:T) this.value = value;
}
private class CommonFixed extends CommonBase<Float> {
    public function new(value:Float) super(value);
}
private class CommonRelay<T> extends CommonBase<T> {
    public function new(value:T) super(value);
}
private class CommonNested<T> extends CommonRelay<Array<T>> {
    public function new(value:Array<T>) super(value);
}
private class CommonFixedArray extends CommonBase<Array<Float>> {
    public function new(value:Array<Float>) super(value);
}

class GenericCommonTypes {
    public static macro function same(actual:Expr, expected:Expr):Expr {
        var a = Context.typeof(actual).toString();
        var b = Context.typeof(expected).toString();
        return macro $v{a == b};
    }
    static function main() {
        var values = [new CommonFixed(1.5), new CommonRelay(2.5)];
        if (!same(values, (null:Array<CommonBase<Float>>))) throw "array common type";
        if (values[0].value != 1.5 || values[1].value != 2.5) throw "array values";
        var selected = if (false) new CommonFixed(1.5) else new CommonRelay(2.5);
        if (!same(selected, (null:CommonBase<Float>))) throw "conditional common type";
        if (selected.value != 2.5) throw "conditional value";
        var switched = switch (1) {
            case 0: new CommonFixed(1.5);
            default: new CommonRelay(2.5);
        };
        if (!same(switched, (null:CommonBase<Float>))) throw "switch common type";
        if (switched.value != 2.5) throw "switch value";
        var nested = [new CommonNested([1.5]), new CommonFixedArray([2.5])];
        if (!same(nested, (null:Array<CommonBase<Array<Float>>>))) throw "nested common type";
        if (nested[0].value[0] != 1.5 || nested[1].value[0] != 2.5) throw "nested values";
        Sys.println("CONFORMANCE_OK");
    }
}
