// Regression: a local initialised from a BITWISE/SHIFT expression must type as
// Int, not Dynamic.
//
// `infer_expression_type` had no arm for & | ^ << >> >>>, so they fell to the
// `_ => Dynamic` catch-all. A local bound to one typed as Dynamic, and any
// later `if (c) v = <const>` then boxed the constant via haxe_box_int_ptr; the
// merge phi degraded to *void and the BOX POINTER was truncated back to i32.
// Symptoms: pointer-shaped values (large, varying run to run) plus a 48-byte
// leak per assignment. Found via f32ToF16's mantissa-carry path returning
// garbage for inputs one ulp below a power of two.
class TestBitwiseConditionalReassign {
    static function shr(x:Int):Int  { var r = x >> 3;  if (x > 0) { r = 0; } return r; }
    static function ushr(x:Int):Int { var r = x >>> 3; if (x > 0) { r = 0; } return r; }
    static function shl(x:Int):Int  { var r = x << 3;  if (x > 0) { r = 0; } return r; }
    static function and(x:Int):Int  { var r = x & 3;   if (x > 0) { r = 0; } return r; }
    static function or(x:Int):Int   { var r = x | 3;   if (x > 0) { r = 0; } return r; }
    static function xor(x:Int):Int  { var r = x ^ 3;   if (x > 0) { r = 0; } return r; }

    // The original shape: reassign, then conditionally reassign. Returns the
    // f16 encoding of a value one ulp below a power of two, which is the only
    // input that reaches the carry branch.
    static function carry(mant:Int, e0:Int):Int {
        var e = e0;
        var r = mant >> 13;
        var rem = mant & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1)) {
            r++;
            if (r == 0x400) { r = 0; e++; if (e >= 31) return 0x7C00; }
        }
        return (e << 10) | r;
    }

    public static function main():Void {
        var fails = 0;
        // Every conditional reassign must yield the assigned constant, not a
        // boxed pointer.
        if (shr(8) != 0)  { fails++; Sys.println("FAIL shr = " + shr(8)); }
        if (ushr(8) != 0) { fails++; Sys.println("FAIL ushr = " + ushr(8)); }
        if (shl(8) != 0)  { fails++; Sys.println("FAIL shl = " + shl(8)); }
        if (and(8) != 0)  { fails++; Sys.println("FAIL and = " + and(8)); }
        if (or(8) != 0)   { fails++; Sys.println("FAIL or = " + or(8)); }
        if (xor(8) != 0)  { fails++; Sys.println("FAIL xor = " + xor(8)); }

        // mant = 0x7FFFFF rounds up and carries: r wraps to 0, e becomes 2.
        var c = carry(8388607, 1);
        if (c != 2048) { fails++; Sys.println("FAIL carry = " + c + " expected 2048"); }
        // No carry: mantissa 0 stays put.
        var c2 = carry(0, 5);
        if (c2 != 5120) { fails++; Sys.println("FAIL carry-no-round = " + c2 + " expected 5120"); }

        // A bitwise result must stay Int through arithmetic, not decay.
        var mask = 0xFF00 & 0xF0F0;
        if (mask != 0xF000) { fails++; Sys.println("FAIL mask = " + mask); }

        Sys.println(fails == 0 ? "PASS bitwise conditional reassign" : ("FAILURES: " + fails));
    }
}
