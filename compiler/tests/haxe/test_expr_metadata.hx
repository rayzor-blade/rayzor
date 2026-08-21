// Haxe metadata is an expression form, not only a declaration one:
// `@:privateAccess obj.field` is EMeta(s, e) in haxe.macro.Expr. The RD parser
// consumed the tokens and threw the metadata away, so `@:move x` or `@:await f`
// could parse and mean nothing. The rest of the pipeline already carried it.
//
// Metadata is transparent to the value: whatever the annotated expression
// evaluates to is what the annotated form evaluates to.
class Box {
    public var v:Int;
    public function new(v:Int) { this.v = v; }
}

class Main {
    static function main():Void {
        var b = new Box(7);

        // On a field access, a call, and a literal.
        Sys.println("field " + @:privateAccess b.v);
        Sys.println("call " + @:privateAccess twice(b.v));
        Sys.println("lit " + @:someMeta 5);

        // Stacked metadata nests; the value is still the inner expression's.
        Sys.println("stacked " + @:outer @:inner b.v);

        // Metadata carrying parameters.
        Sys.println("params " + @:tag("a", 2) b.v);

        // As a call argument, and inside arithmetic.
        Sys.println("arg " + twice(@:m b.v));
        Sys.println("arith " + (@:m b.v + 1));

        // Annotated expressions keep their type: this must stay an Int.
        var n:Int = @:m b.v;
        Sys.println("typed " + (n * 3));

        if (@:m b.v == 7) Sys.println("cond ok"); else Sys.println("FAIL cond");
    }

    static function twice(x:Int):Int { return x * 2; }
}
