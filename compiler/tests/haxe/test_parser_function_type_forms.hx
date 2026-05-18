// Regression test for compiler fix #2: parser accepts every standard
// Haxe function-type syntactic shape, not just the curried form.
//
// Pre-fix, only `A->B->C` (curried) parsed. The parameterised forms
// `(name:A, name:B)->C`, `(A, B)->C`, and `()->C` all failed with
// "expected ')' to close parameter list" because parse_basic_type
// treated the leading `(` as a parenthesised single type and got
// confused by the comma / name colon.
//
// After: types.rs parses a comma-separated list of optionally-named
// type entries inside `(...)`, then disambiguates by whether `->`
// follows — function type if yes, parenthesised single type otherwise.

class Box {
    public function new() {}

    public function curried(fn:Int->Int->Void):Void { fn(1, 2); }
    public function named(fn:(a:Int, b:Int)->Void):Void { fn(3, 4); }
    public function unnamed(fn:(Int, Int)->Void):Void { fn(5, 6); }
    public function nullary(fn:()->Void):Void { fn(); }
}

class Main {
    static function main() {
        var b = new Box();
        b.curried(function(a:Int, b:Int):Void { trace("curried:" + a + "," + b); });
        b.named(function(a:Int, b:Int):Void { trace("named:" + a + "," + b); });
        b.unnamed(function(a:Int, b:Int):Void { trace("unnamed:" + a + "," + b); });
        b.nullary(function():Void { trace("nullary"); });
    }
}
