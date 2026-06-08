typedef Probe = {
    var s:String;
    var n:Int;
    var f:Float;
    var b:Bool;
}

class Main {
    static function makeProbe():Probe {
        var s = "llama";
        var n = 2048;
        var f = 3.5;
        var b = true;
        return {
            s: s,
            n: n,
            f: f,
            b: b
        };
    }

    static function main() {
        var p = makeProbe();
        if (p.s != "llama") throw "bad string field";
        if (p.n != 2048) throw "bad int field";
        if (p.f != 3.5) throw "bad float field";
        if (!p.b) throw "bad bool field";
        trace(p.s);
        trace(p.n);
        trace(p.f);
        trace(p.b);
    }
}
