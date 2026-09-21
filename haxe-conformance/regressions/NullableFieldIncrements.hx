// A `Null<T>` field of a generic class is a box slot: a scalar written into
// it is boxed, and `++`/`--` on any `Null<scalar>` opens the box, counts and
// closes it. A null nested enum payload matches no constructor pattern.
private class Foo<T> {
    public var val:T;
    public var valNull:Null<T>;
    public function new() {}
}
private enum T { T1(x:Null<T>); T2(x:Int); }
class NullableFieldIncrements {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function match(t) {
        return switch (t) {
            case T1(T1(_)): 0;
            case T1(T2(_)): 1;
            case T1(null): 2;
            case T2(_): 3;
        }
    }
    static function main() {
        var f = new Foo<Int>();
        f.val = 0;
        check("erased field", (f.val++) + " " + f.val + " " + (++f.val) + " " + (f.val += 10), "0 1 2 12");
        f.valNull = 0;
        check("nullable field", Std.string(f.valNull) + " " + (f.valNull++) + " " + f.valNull + " " + (++f.valNull) + " " + (f.valNull += 10), "0 0 1 2 12");
        var n:Null<Int> = 0;
        check("nullable local", (n++) + " " + n + " " + (++n) + " " + (n += 10) + " " + n, "0 1 2 12 12");
        var m:Null<Int> = 5; m++; m += 2; m--;
        check("nullable statements", Std.string(m), "7");
        var a:Array<Null<Int>> = [1]; a[0]++;
        check("nullable element", Std.string(a[0]), "2");
        check("nested null pattern", match(T1(T1(null))) + " " + match(T1(T2(1))) + " " + match(T1(null)) + " " + match(T2(2)), "0 1 2 3");
        Sys.println("CONFORMANCE_OK");
    }
}
