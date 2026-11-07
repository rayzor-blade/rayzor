import p.A;
import p.B;
import p.C;
import p.D;
import p.F;
import p.E;

class CyclicStaticReads {
    static function main() {
        var a = new A();
        var b = new B();
        a.link(b);
        if (a.peers() != 11) throw "A reads B.total: " + a.peers();
        if (b.size() != 11) throw "B reads A.count: " + b.size();
        if (b.mode(Fast) != "fast") throw "enum from the cyclic module";
        b.zero();
        if (A.count != 0) throw "B writes A.count: " + A.count;
        // Each module's own initializers survive when the other reads its statics.
        if (D.total != 10 || C.count != 1) throw "initializers in a cycle: " + D.total + " " + C.count;
        if (C.peers() != 11 || D.size() != 11) throw "static methods in a cycle: " + C.peers() + " " + D.size();
        if (F.total != 20 || E.count != 2) throw "initializers in a reversed cycle: " + F.total + " " + E.count;
        if (E.peers() != 22 || F.size() != 22) throw "static methods in a reversed cycle: " + E.peers() + " " + F.size();
        trace("CONFORMANCE_OK");
    }
}
