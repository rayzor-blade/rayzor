// Custom annotations: a macro reads `@:name(params) expr` off the expression it
// was handed. The reified form is EMeta(s:MetadataEntry, e:Expr), matching
// haxe.macro.Expr, with MetadataEntry = { name, params, pos }.
//
// KNOWN LIMIT, asserted below so it is visible rather than folded away: the
// annotation must be the OUTERMOST node of the macro's argument. Parenthesising
// it hides it, because EParenthesis is not reified and the walk stops at an
// opaque Expr.
import haxe.macro.Expr;

class Ann {
    macro static function nameOf(e:haxe.macro.Expr):haxe.macro.Expr {
        switch (e.expr) {
            case EMeta(s, _): return macro $v{s.name};
            case _: return macro "none";
        }
    }

    macro static function arity(e:haxe.macro.Expr):haxe.macro.Expr {
        switch (e.expr) {
            case EMeta(s, _): return macro $v{s.params.length};
            case _: return macro -1;
        }
    }

    /** Drop the annotation, keep what it annotated. */
    macro static function unwrap(e:haxe.macro.Expr):haxe.macro.Expr {
        switch (e.expr) {
            case EMeta(_, inner): return inner;
            case _: return e;
        }
    }
}

class Main {
    static function main():Void {
        eq("name", Ann.nameOf(@:mine 1), "mine");
        eq("name.other", Ann.nameOf(@:other 1), "other");
        eq("name.bare", Ann.nameOf(1), "none");

        eqi("arity.0", Ann.arity(@:tag 1), 0);
        eqi("arity.1", Ann.arity(@:tag("a") 1), 1);
        eqi("arity.2", Ann.arity(@:tag("a", "b") 1), 2);
        eqi("arity.bare", Ann.arity(1), -1);

        // The annotated expression survives, annotation removed.
        eqi("unwrap", Ann.unwrap(@:whatever 20 + 3), 23);

        // Stacked metadata nests; the outermost is what a single match sees.
        eq("stacked", Ann.nameOf(@:outer @:inner 1), "outer");

        // Documented limit -- parenthesised metadata is not visible yet.
        eq("parens.limit", Ann.nameOf((@:mine 1)), "none");

        Sys.println("macro annotations ok");
    }

    static function eq(tag:String, got:String, want:String):Void {
        if (got != want) Sys.println("FAIL " + tag + ": got " + got + " want " + want);
    }
    static function eqi(tag:String, got:Int, want:Int):Void {
        if (got != want) Sys.println("FAIL " + tag + ": got " + got + " want " + want);
    }
}
