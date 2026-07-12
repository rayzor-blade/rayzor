import rayzor.concurrent.Channel;

// Uniform-boxing Channel<T>: primitives round-trip through the DynamicValue box
// (incl. the zero/false edge cases), and the null-safety crux holds — a sent 0
// is a non-null box that unboxes to 0, distinct from an empty channel (null).
class Main {
    static function main() {
        var ci = new Channel<Int>(4);
        ci.send(0);
        ci.send(7);
        var i0:Int = ci.receive();
        if (i0 != 0) throw "int 0 roundtrip got " + i0;
        var i7:Int = ci.receive();
        if (i7 != 7) throw "int 7 roundtrip got " + i7;

        // Inferred channel receive() — T erases to i64 (the same-thread bug shape).
        var inf = new Channel(4);
        inf.send(42);
        var iv:Int = inf.receive();
        if (iv != 42) throw "inferred int receive got " + iv;

        var cf = new Channel<Float>(4);
        cf.send(0.0);
        cf.send(2.5);
        var f0:Float = cf.receive();
        if (f0 != 0.0) throw "float 0.0 roundtrip got " + f0;
        var f25:Float = cf.receive();
        if (f25 != 2.5) throw "float 2.5 roundtrip got " + f25;

        var cb = new Channel<Bool>(4);
        cb.send(false);
        cb.send(true);
        var bf:Bool = cb.receive();
        if (bf != false) throw "bool false roundtrip got " + bf;
        var bt:Bool = cb.receive();
        if (bt != true) throw "bool true roundtrip got " + bt;

        // Null-safety crux: a sent 0 is a non-null box (distinct from an empty
        // channel, which is null). Value round-trip of 0 is covered by the
        // receive() checks above; here we assert the null distinction.
        var cz = new Channel<Int>(4);
        cz.send(0);
        var z:Null<Int> = cz.tryReceive();
        if (z == null) throw "sent 0 read as null (boxed 0 must be non-null)";
        var e:Null<Int> = cz.tryReceive();
        if (e != null) throw "empty channel tryReceive was not null";

        Sys.println("PASS channel-prim-roundtrip");
    }
}
