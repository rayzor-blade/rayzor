class BackendFrames {
    static function main() {
        var sum = 0;
        for (i in 0...30) {
            var even = i % 2 == 0;
            for (j in 0...3) if (even) sum += j;
        }
        if (sum != 45) throw "OSR primitive frame slots";
        var errors = 0;
        function equal<T>(a:T, b:T) { if (a != b) errors++; }
        var check = function() equal(1, 1);
        check();
        if (errors != 0) throw "erased closure arguments";
        trace("CONFORMANCE_OK");
    }
}
