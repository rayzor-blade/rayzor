// A comprehension body's `if` is a filter: the push moves into each branch,
// so an iteration no branch yields a value for adds nothing. An `if`/`else`
// statement's else arm starts from the locals as they were before the `if`.
class ComprehensionFilters {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function main() {
        var r = [for (i in 0...10) if (i % 2 == 0) i else if (i % 3 == 0) i];
        check("else-if filter", r.length + ":" + r.join(","), "7:0,2,3,4,6,8,9");
        check("plain filter", [for (i in 0...6) if (i % 2 == 0) i].join(","), "0,2,4");
        check("both arms", [for (i in 0...4) if (i > 1) "a" + i else "b"].join(","), "b,b,a2,a3");
        check("block tail", [for (i in 0...5) { var d = i * 2; if (d > 4) d; }].join(","), "6,8");
        check("nested", [for (i in 0...3) for (j in 0...3) if (i == j) i * 10 + j].join(","), "0,11,22");
        trace("CONFORMANCE_OK");
    }
}
