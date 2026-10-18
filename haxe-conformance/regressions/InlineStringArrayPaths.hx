// charCodeAt on ASCII and non-ASCII strings, and push onto arrays of every
// element kind, must give the same results whether done in place or by the
// runtime.
class P {
    public var v:Int;

    public function new(v)
        this.v = v;
}

class Main {
    static function main() {
        var a = "hello";
        var u = "héllo€";
        var e = "";
        var codes:Array<Null<Int>> = [
            a.charCodeAt(0), a.charCodeAt(4), a.charCodeAt(5), a.charCodeAt(-1),
            u.charCodeAt(1), u.charCodeAt(5), u.charCodeAt(6), e.charCodeAt(0)
        ];
        if (codes.join(",") != "104,111,null,null,233,8364,null,null") throw "charCodeAt " + codes.join(",");
        var built = new StringBuf();
        for (i in 0...3)
            built.addChar(65 + i);
        var b = built.toString();
        if (b.charCodeAt(2) != 67 || b.charCodeAt(3) != null) throw "built string";
        var sum = 0;
        for (i in 0...a.length)
            sum += a.charCodeAt(i);
        if (sum != 532) throw "sum";

        var ints:Array<Int> = [];
        for (i in 0...100)
            ints.push(i * 3 - 7);
        var n = 0;
        for (x in ints)
            n += x;
        if (n != 14150 || ints.push(5) != 101 || ints.length != 101) throw "Int push";
        var floats:Array<Float> = [1.5];
        for (i in 0...20)
            floats.push(i / 4);
        var fs = 0.0;
        for (x in floats)
            fs += x;
        if (fs != 49) throw "Float push";
        var dyn:Array<Dynamic> = [1];
        for (i in 0...10)
            dyn.push(i % 2 == 0 ? (i : Dynamic) : ("s" + i : Dynamic));
        if (dyn.join("|") != "1|0|s1|2|s3|4|s5|6|s7|8|s9") throw "Dynamic push";
        var strs = ["x"];
        for (i in 0...5)
            strs.push("y" + i);
        if (strs.join(",") != "x,y0,y1,y2,y3,y4") throw "String push";
        var objs = [new P(1)];
        for (i in 0...9)
            objs.push(new P(i));
        var os = 0;
        for (p in objs)
            os += p.v;
        if (os != 37) throw "object push";
        var nulls:Array<Null<Int>> = [null];
        for (i in 0...4)
            nulls.push(i == 2 ? null : i);
        if (nulls.join(",") != "null,0,1,null,3") throw "Null<Int> push";
        Sys.println("CONFORMANCE_OK");
    }
}
