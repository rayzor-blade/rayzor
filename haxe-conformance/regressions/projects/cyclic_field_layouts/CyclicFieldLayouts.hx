import p.Tree;
import p.Scroll;
import p.Bounds;

class CyclicFieldLayouts {
    static function main() {
        var t = new Tree(4, 2);
        t.scroll = new Scroll(t, 10, 20);
        if (t.scrolled() != 30) throw "scrolled " + t.scrolled();
        if (t.scroll.owner() != 204) throw "owner " + t.scroll.owner();
        var b = new Bounds(1, 2);
        var m = new p.Bounds.Marker();
        if (b.ptr != 3 || m.ptr != 9 || m.x != 7 || m.id != 3) throw "other layouts";
        trace("CONFORMANCE_OK");
    }
}
