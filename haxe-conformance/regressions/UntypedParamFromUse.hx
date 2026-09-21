// A parameter the source leaves unannotated takes the type of the annotated
// formal it is passed to, the way Haxe unifies the monomorph. Left Dynamic,
// the caller boxed the argument and the callee read the box as the value.
// The recovery is recorded when the class's signatures are registered, so a
// caller lowered before the callee's body agrees with it.
private typedef Tok = { var p:String; var s:Bool; }
class UntypedParamFromUse {
    var x:Int;
    function new() { x = 0; }
    static function typedLen(l:List<Tok>):Int return l.length;
    static function fwd(l) return typedLen(l);
    static function fwd2(l) return fwd(l);           // resolves through fwd on a later round
    static function addOne(x:Int):Int return x + 1;
    static function fwdInt(x) return addOne(x);
    static function strLen(s:String):Int return s.length;
    static function fwdStr(s) return strLen(s);
    static function arrLen(a:Array<Int>):Int return a.length;
    static function fwdArr(a) return arrLen(a);
    // Operator uses type a parameter the way Haxe unifies the monomorph.
    static function sub(from, to) return to - from;
    static function idx(a:Array<Int>, from) return a[from];
    static function lt(from, to) return from < to;
    static function mid(from, to) { var m = (from + to) >> 1; return m; }
    static function loop(from, to) { var n = 0; var i = from; while (i < to) { n += i; i++; } return n; }
    static function fwdBoth(a:Array<Int>, from, to) return sub(from, to) + idx(a, from);
    static function cat(s) return "a" + s;
    static function asString(x):String return x;
    static function asBool(x) { var y:Bool = x; return y; }
    static function dflt(a = 2, b = 4.25) return "" + a + "/" + b;
    static function main() {
        var l = new List<Tok>(); l.add({p: "x", s: true});
        if (fwd(l) != 1) throw "List through an untyped parameter";
        if (fwd2(l) != 1) throw "List through two untyped parameters";
        if (fwdInt(41) != 42) throw "Int through an untyped parameter";
        if (fwdStr("abc") != 3) throw "String through an untyped parameter";
        if (fwdArr([1, 2, 3]) != 3) throw "Array through an untyped parameter";
        var o = new UntypedParamFromUse();
        if (o.set(7) != 7) throw "field-store parameter, caller above the callee";
        if (twice(21) != 42) throw "call-use parameter, caller above the callee";
        // An explicitly Dynamic argument to a scalar formal is unboxed at the call.
        var d:Dynamic = 41;
        if (addOne(d) != 42) throw "Dynamic argument to an Int formal";
        if (sub(1, 4) != 3) throw "arithmetic operand";
        if (idx([7, 8, 9], 1) != 8) throw "array index";
        if (!lt(1, 4) || lt(4, 1)) throw "comparison operand";
        if (mid(2, 6) != 4) throw "bit-shift operand";
        if (loop(0, 4) != 6) throw "copied into a local, then compared and incremented";
        if (fwdBoth([7, 8, 9], 1, 4) != 11) throw "forwarded to two inferred callees";
        if (cat("x") != "ax") throw "string concatenation";
        if (asString("5") != "5") throw "returned through an annotated return type";
        if (!asBool(true)) throw "stored into an annotated local";
        if (dflt() != "2/4.25" || dflt(5, 8.5) != "5/8.5") throw "defaulted parameters keep the default's type";
        trace("CONFORMANCE_OK");
    }
    function set(v) { this.x = v; return this.x; }
    static function twice(v) { return addOne(v) * 2 - 2; }
}
