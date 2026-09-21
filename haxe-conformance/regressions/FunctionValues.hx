// A function used as a value has the shape of a lambda of its type, so a
// call through a typed function value reads the declared result; a call
// through a Dynamic-typed value reaches the closure's box-shaped entry; the
// runtime's array helpers reach its slot-shaped entry; a function literal
// called in place types its parameters from the arguments.
class FunctionValues {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function compare(a:Int, b:Int):Int return a - b;
    static function half(x:Float):Float return x / 2;
    static function shout(s:String):String return s.toUpperCase() + "!";
    static function isOdd(n:Int):Bool return n % 2 == 1;
    static function apply(f:Int->Int->Int, x:Int, y:Int):Int return f(x, y);
    static function applyDyn(f, x, y) return f(x, y);
    static function applyU<T>(a:Array<T>, f, i, j) return f(a[i], a[j]);
    static function callTyped(cmp:Int->Int->Int, x, y) return cmp(x, y);
    static function describe(f:Dynamic, s:String):String return f(s);
    static function applyT<T>(f:T->T->Int, x:T, y:T):Int return f(x, y);
    static function blub(o:{counter:Int}, increment:Bool) {
        if (increment) o.counter++; else o.counter--;
        return o.counter;
    }
    static function main() {
        check("typed", apply(compare, 5, 2) + " " + apply((a, b) -> a - b, 5, 2), "3 3");
        // Through a Dynamic-typed value: the closure's box-shaped entry.
        check("dynamic", applyDyn(compare, 5, 2) + " " + applyDyn((a, b) -> a - b, 5, 2), "3 3");
        check("dynamic untyped formal", applyU([5, 2], compare, 0, 1) + " " + applyU([5, 2], (a, b) -> a - b, 0, 1), "3 3");
        check("dynamic args to typed formal", callTyped(compare, 5, 2) + " " + callTyped((a, b) -> b - a, 5, 2), "3 -3");
        check("dynamic string", describe(shout, "hey") + " " + describe(s -> s + "?", "hey"), "HEY! hey?");
        var hd:Dynamic = half;
        check("dynamic float", hd(5) + " " + hd(2.5), "2.5 1.25");
        check("generic", applyT(compare, 5, 2) + " " + applyT((a, b) -> a - b, 5, 2) + " " + applyT((a, b) -> a.length - b.length, "hello", "hi"), "3 3 3");
        var names = ["b", "a", "c"]; names.sort((x, y) -> x < y ? -1 : 1);
        var names2 = ["b", "a", "c"]; haxe.ds.ArraySort.sort(names2, (x, y) -> x < y ? -1 : 1);
        check("generic lambda over strings", names.join("") + names2.join(""), "abcabc");
        var f = compare;
        check("local", f(5, 2) + " " + apply(f, 5, 2) + " " + applyT(f, 5, 2), "3 3 3");
        var h = half, s = shout, o = isOdd;
        check("kinds", h(5) + " " + s("hey") + " " + o(3) + " " + o(4), "2.5 HEY! true false");
        check("map", [1, 2, 3].map(isOdd).join(",") + " " + ["a", "b"].map(shout).join(",") + " " + [2.0, 4.0].map(half).join(",") + " " + [2.0, 4.0].map(x -> x / 2).join(","), "true,false,true A!,B! 1,2 1,2");
        check("filter", [1, 2, 3, 4].filter(isOdd).join(","), "1,3");
        var arr = [5, 2, 9, 1]; arr.sort(compare);
        var arr2 = [5, 2, 9, 1]; haxe.ds.ArraySort.sort(arr2, compare);
        check("sort", arr.join(",") + " " + arr2.join(","), "1,2,5,9 1,2,5,9");
        var o_blub = (function(f, o) {
            return function(increment) return f(o, increment);
        })(blub, {counter: 0});
        check("iife", o_blub(true) + " " + o_blub(true) + " " + o_blub(false), "1 2 1");
        Sys.println("CONFORMANCE_OK");
    }
}
