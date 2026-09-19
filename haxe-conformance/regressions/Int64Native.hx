// Int64 is the native i64 here, and the abstract's own methods must agree
// with that: they read `this.high` / `this.low` on the two-word class the
// abstract declares over, which took a slot GEP on the i64 (isNeg faulted,
// divMod never ran, toStr printed ""). A qualified `haxe.Int64.make` reaches
// the abstract like `Int64.make` does, and `Int32 + String` is a concat, not
// the abstract's @:op(A + B).
import haxe.Int64;
class Int64Native {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function main() {
        var a = Int64.make(0, 47);
        check("make/toStr", Int64.toStr(a), "47");
        check("qualified make", Int64.toStr(haxe.Int64.make(0, 5)), "5");
        check("ofInt", Int64.toStr(Int64.ofInt(7)), "7");
        check("isNeg", Std.string(a.isNeg()), "false");
        var ten:Int64 = 10;
        check("implicit from Int", Std.string(ten.low), "10");
        var r = Int64.divMod(a, ten);
        check("divMod q", Std.string(r.quotient.low), "4");
        check("divMod m", Std.string(r.modulus.low), "7");
        check("neg", Int64.toStr(Int64.neg(a)), "-47");
        check("Std.string", Std.string(a), "47");
        check("big", Int64.toStr(Int64.make(1, 0)), "4294967296");
        check("negative", Int64.toStr(Int64.make(-1, -1)), "-1");
        var x:haxe.Int32 = 7;
        var s = "";
        s = x + s;
        check("Int32 + String", s, "7");
        Sys.println("CONFORMANCE_OK");
    }
}
