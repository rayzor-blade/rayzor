class TestGuard {
    static function classify(x:Int):String {
        // Expression-position switch with guards
        return switch (x) {
            case v if (v > 100): "huge";
            case v if (v > 10): "big";
            case v if (v > 0): "small";
            case 0: "zero";
            default: "negative";
        };
    }

    static function main() {
        // Test expression-position guards
        trace(classify(200));
        trace(classify(50));
        trace(classify(5));
        trace(classify(0));
        trace(classify(-3));

        // Test statement-position guards
        var y = 42;
        switch (y) {
            case v if (v > 100): trace("over100");
            case v if (v > 10): trace("over10");
            default: trace("small");
        }
    }
}
