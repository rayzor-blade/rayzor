// A parameter assigned inside an `if` (the `?len:Int` null-default idiom)
// must read as the joined value afterwards: the join built the phi but
// left the parameter bound to the last branch's register (a parameter has
// no `locals` entry, and only locals were rebound).
class ParamReassignInBranch {
    static function dflt(?len:Int):Int { if (len == null) len = 10; return len; }
    static function both(?len:Int):Int { if (len == null) { len = 10; } else { len = 20; } return len; }
    static function plain(len:Int):Int { if (len == 0) len = 10; return len; }
    static function expr(?len:Int):Int { var r = if (len == null) { len = 5; len } else { len = len * 2; len }; return r + len; }
    static function main() {
        if (dflt() != 10) throw "dflt()";
        if (dflt(3) != 3) throw "dflt(3)";
        if (both() != 10 || both(1) != 20) throw "both";
        if (plain(0) != 10 || plain(4) != 4) throw "plain";
        if (expr() != 10 || expr(3) != 12) throw "expr";
        Sys.println("CONFORMANCE_OK");
    }
}
