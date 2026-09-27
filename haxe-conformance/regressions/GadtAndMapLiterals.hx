// An enum constructor may name its own result type (`P:C<PR>`) and still is
// a constructor a case matches by. An un-annotated map literal is the same
// concrete map `new Map<K, V>()` builds, and every runtime map has copy().
private class PR {}
private class VR {}
private enum C<X> { P:C<PR>; V:C<VR>; }
typedef Td = E<String>;
private enum E<T> { K(s:String):Td; }
class GadtAndMapLiterals {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function which<X>(c:C<X>):String {
        return switch c { case P: "P"; case _: "other"; }
    }
    static function unwrap<T>(e:E<T>):String {
        return switch e { case K(s): s; }
    }
    static function main() {
        check("gadt case", which(P) + " " + which(V), "P other");
        check("gadt ctor", unwrap(K("foo")), "foo");

        var lit = [1 => "x", 2 => "y"];
        check("literal get", lit.get(1), "x");
        var lc = lit.copy();
        lc.remove(1);
        check("int copy", lc.exists(1) + " " + lit.exists(1) + " " + lc.get(2), "false true y");

        var sm = new Map<String, Int>();
        sm.set("a", 1);
        var sc = sm.copy();
        sc.set("a", 10);
        check("string copy", sm.get("a") + " " + sc.get("a"), "1 10");
        trace("CONFORMANCE_OK");
    }
}
