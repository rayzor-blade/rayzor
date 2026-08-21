// SIMD4f.round() must give the same answer at every tier. The interesting
// inputs are exact ties, where round-half-to-even and round-half-away-from-zero
// disagree: 0.5 -> 0 vs 1, 2.5 -> 2 vs 3.
//
// Ties-to-even is the contract, because it is the IEEE-754 default and the one
// mode every target reaches in a single instruction. LLVM alone used to emit
// llvm.round (ties-away), so the same function changed value on tier-up.
import rayzor.SIMD4f;

class Main {
    static function main():Void {
        check("pos", SIMD4f.make(0.5, 1.5, 2.5, 3.5).round(), 0.0, 2.0, 2.0, 4.0);
        check("neg", SIMD4f.make(-0.5, -1.5, -2.5, -3.5).round(), -0.0, -2.0, -2.0, -4.0);
        // Non-ties must be unaffected by the change of rounding mode.
        check("frac", SIMD4f.make(0.4, 0.6, -0.4, -0.6).round(), 0.0, 1.0, -0.0, -1.0);
    }

    static function check(tag:String, v:SIMD4f, a:Float, b:Float, c:Float, d:Float):Void {
        var want = [a, b, c, d];
        for (i in 0...4) {
            if (v.get(i) != want[i]) {
                Sys.println("FAIL " + tag + " lane " + i + ": got " + v.get(i) + " want " + want[i]);
            }
        }
        Sys.println(tag + " " + v.get(0) + " " + v.get(1) + " " + v.get(2) + " " + v.get(3));
    }
}
