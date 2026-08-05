class Main {
    static function main() {
        // 2^24 = 16777216. In TRUE f32, 16777216 + 1 == 16777216 (unrepresentable).
        // If the adds were done in f64 and only rounded at the end, ten +1s would
        // reach 16777226. This distinguishes per-step f32 from f64-then-round.
        var s:Single = 16777216.0;
        for (i in 0...10) s = s + 1.0;
        var fs:Float = s;

        var d:Float = 16777216.0;
        for (i in 0...10) d = d + 1.0;

        // Literal narrowing must round to the nearest f32, and f32 must
        // overflow where f64 does not.
        var lit:Single = 1.2345678901234567;
        var litF:Float = lit;
        var big:Single = 1e300;
        var bigF:Float = big;

        var ok = fs == 16777216.0 && d == 16777226.0
            && litF == 1.2345678806304932 && bigF > 1e308;
        if (ok) {
            Sys.println("PASS single-f32 perStep=" + fs + " f64=" + d + " lit=" + litF);
        } else {
            Sys.println("FAIL single-f32 perStep=" + fs + " f64=" + d
                + " lit=" + litF + " big=" + bigF);
        }
    }
}
