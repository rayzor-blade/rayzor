// Regression: SIMD4f arithmetic METHODS (a.add(b)/.sub/.mul/.div) must produce
// the same result as the OPERATORS (a + b ...). The method-call path routed them
// to a MIR wrapper that mishandled the vector ABI and returned garbage (or a
// I64->Vector cast trap in a loop-phi); they now lower to VectorBinOp directly.
import rayzor.SIMD4f;
class Main {
    static function eq(got:Float, want:Float, msg:String):Void {
        var d = got - want; if (d < 0) d = -d;
        if (d > 0.01) { trace("FAIL " + msg + ": got " + got + " want " + want); Sys.exit(1); }
    }
    static function main() {
        var a = SIMD4f.make(1.0, 2.0, 3.0, 4.0);
        var b = SIMD4f.make(2.0, 2.0, 2.0, 2.0);
        // methods vs operators must agree
        eq(a.add(b).sum(), (a + b).sum(), "add==+");
        eq(a.sub(b).sum(), (a - b).sum(), "sub==-");
        eq(a.mul(b).sum(), (a * b).sum(), "mul==*");
        eq(a.div(b).sum(), (a / b).sum(), "div==/");
        // absolute values
        eq(a.add(b).sum(), 18.0, "add");
        eq(a.mul(b).sum(), 20.0, "mul");
        eq(a.dot(b), 20.0, "dot");
        eq(a.sum(), 10.0, "sum");
        // method in a loop-carried accumulator (was a W0020 I64->Vector trap)
        var acc = SIMD4f.splat(0.0);
        var i = 0;
        while (i < 1000) { acc = acc.add(a); i++; }
        eq(acc.sum(), 10000.0, "loop add accumulate");
        trace("test_simd4f_methods OK");
    }
}
