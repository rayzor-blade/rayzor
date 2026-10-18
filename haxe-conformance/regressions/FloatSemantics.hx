// Float operations round one at a time, and NaN, the infinities and signed
// zeros are ordinary values that comparisons and arithmetic must respect.
class Main {
    static function main() {
        var zero = Std.parseFloat("0");
        var tenth = Std.parseFloat("0.1");
        var nan = zero / zero;
        var inf = 1 / zero;
        var negZero = -zero;

        // Fused, this is the rounding error of 0.1 (about 5.6e-17).
        if (tenth * 10 - 1 != 0) throw "multiply-add was fused";

        if (!(nan != nan)) throw "NaN != NaN";
        if (nan == nan) throw "NaN == NaN";
        if (!Math.isNaN(nan)) throw "isNaN";
        if (nan < 1 || nan >= 1) throw "NaN ordered";
        if (!(inf > 1e308)) throw "infinity";
        if (Math.isFinite(inf)) throw "isFinite";
        if (!Math.isNaN(inf - inf)) throw "inf - inf";
        if (1 / negZero != -inf) throw "1 / -0";
        if (1 / (negZero + 0.0) != inf) throw "-0 + 0";
        Sys.println("CONFORMANCE_OK");
    }
}
