private class P {
    public var a:Int;
    public var s:String;
    public function new(a:Int) { this.a = a; s = "p" + a; }
}
class NullablePathFactory {
    static var flag = true;
    static function maybe(x:Int):Null<P> {
        if (!flag) return null;
        return new P(x);
    }
    static function maybeArray(x:Int):Null<Array<Int>> {
        if (x < 0) return null;
        return [x, x + 1];
    }
    static function maybeString(x:Int):Null<String> {
        if (x < 0) return null;
        return "v" + x;
    }
    static function main() {
        var sum = 0;
        for (i in 0...1000) {
            var p = maybe(i);
            if (p != null) sum += p.a;
            var arr = maybeArray(i - 500);
            if (arr != null) sum += arr[1];
            var s = maybeString(i - 500);
            if (s != null) sum += s.length;
        }
        if (sum != 499500 + 125250 + sumLengths()) throw "sum " + sum;
        flag = false;
        if (maybe(1) != null) throw "null path";
        if (maybeArray(-1) != null) throw "null array path";
        if (maybeString(-1) != null) throw "null string path";
        Sys.println("CONFORMANCE_OK");
    }
    static function sumLengths():Int {
        var t = 0;
        for (i in 500...1000) t += ("v" + (i - 500)).length;
        return t;
    }
}
