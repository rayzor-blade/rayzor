// Haxe's precedence ladder, where it differs from C: the bitwise operators
// bind tighter than the comparisons (`n & 0x8000 != 0` is `(n & 0x8000) != 0`;
// both parsers read it the other way), `...` sits below them, `%` above `*`.
class OperatorPrecedence {
    static function check(label:String, got:Bool) { if (!got) throw label; }
    static function main() {
        var n = 7;
        check("& before !=", (n & 0x8000 != 0) == false);
        check("& before ==", (n & 1 == 1) == true);
        check("| before ==", (n | 8 == 15) == true);
        check("^ before ==", (n ^ 1 == 6) == true);
        check("&& below comparisons", (1 + 2 == 3 && 4 & 4 == 4) == true);
        check(">> before ==", (n >> 1 == 3) == true);
        check("| and & one level, left assoc", (1 | 2 & 3) == 3);
        check("|| still logical", (n == 7 || n & 1 == 0) == true);
        var range = [for (i in 1 + 1...2 * 2) i];
        check("... below arithmetic", range.length == 2 && range[0] == 2);
        check("% above *", 7 % 4 * 2 == 6);
        check("metadata before addition", 5 * @foo 3 + 4 == 19);
        check("nested metadata", 5 * @foo @bar 3 + @baz 4 == 19);
        check("metadata keeps parentheses", 5 * @foo (3 + 4) == 35);
        check("metadata with unary", -@foo 3 + 4 == 1);
        check("metadata on ternary condition", (@foo true ? 3 : 4) == 3);
        var assigned = 0;
        @foo assigned = 3;
        @foo @bar assigned += 4;
        check("annotated assignment", assigned == 7);
        Sys.println("CONFORMANCE_OK");
    }
}
