// `for (c in s)` over a String yields char codes. It went down the array path
// and read the string's bytes as elements.
class StringIteration {
    static function count(s:String):Int { var n = 0; for (c in s) if (c != " ".code) n++; return n; }
    static function main() {
        var sum = 0;
        for (c in "abc") sum += c;
        if (sum != 294) throw "literal: " + sum;
        if (count("  x") != 1) throw "parameter";
        var p = "  xy";
        if (count(p.substr(1)) != 2) throw "substr";
        var kv = "";
        for (i => c in " x") kv += i + ":" + c + " ";
        if (kv != "0:32 1:120 ") throw "key-value: " + kv;
        trace("CONFORMANCE_OK");
    }
}
