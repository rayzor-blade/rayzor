class ERegConstructor {
    static function main() {
        var r = new EReg("b", "");
        if (!r.matchSub("aba", 0, -1)) throw "matchSub";
        if (r.matched(0) != "b") throw "matched";
        var g = new EReg("a", "g");
        if (g.replace("aba", "x") != "xbx") throw "global replace";
        Sys.println("CONFORMANCE_OK");
    }
}
