import p.A;
import p.B;

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
        trace("CONFORMANCE_OK");
    }
}
