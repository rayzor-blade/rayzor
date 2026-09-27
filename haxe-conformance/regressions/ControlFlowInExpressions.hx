// return/break/continue/throw inside an expression leave it: nothing after
// them runs, and the function's return type comes from such a return too.
// A condition's side effects are seen by both arms of the if it guards.
class ControlFlowInExpressions {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function inCall() { var ans = Std.string(return "r") + ""; }
    static function inArray() { var ans = [Std.string(return "r")][0]; }
    static function inThrow() { var ans = Std.string(throw "t") + ""; }
    static function withBreak() {
        while (true) { var ans = Std.string(break) + ""; return "b"; }
        return "a";
    }
    static function withContinue() {
        var a = 0;
        while (true) {
            if (a++ > 0) return "a";
            var ans = Std.string(continue) + "";
            return "b";
        }
    }
    static function inInterpolation(n:Int) {
        return '$n cell${if (n == 1) {return "";} else {return "s";}}.';
    }
    static function main() {
        check("return in call", inCall(), "r");
        check("return in array", inArray(), "r");
        var threw = try { inThrow(); "no"; } catch (e:String) e;
        check("throw in call", threw, "t");
        check("break", withBreak(), "a");
        check("continue", withContinue(), "a");
        check("interpolation", inInterpolation(1) + inInterpolation(2), "s");

        var a = 0;
        if (a++ > 5) a = 100;
        check("if condition effect", "" + a, "1");
        var b = 0;
        var t = (b++ > 5) ? 1 : 2;
        check("ternary condition effect", b + " " + t, "1 2");
        var c = 0;
        if (c++ > 5) c = 100 else c += 10;
        check("else sees condition effect", "" + c, "11");
        var d = 0;
        for (i in 0...3) if (d++ > 5) d = 100;
        check("loop condition effect", "" + d, "3");
        trace("CONFORMANCE_OK");
    }
}
