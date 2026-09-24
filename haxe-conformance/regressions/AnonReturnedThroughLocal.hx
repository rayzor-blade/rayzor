// An un-annotated method returning a local initialised with an object
// literal returns that structure, even to a field initialiser typed before
// the method body; elements of an array of structures are the objects.
private class Holder {
    extern inline function make() {
        var s = {i: 0};
        return s;
    }
    public var a = [make(), make(), make()];
    public function new() {}
}
class AnonReturnedThroughLocal {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function main() {
        var h = new Holder();
        check("distinct", "" + (h.a[0] == h.a[1]) + " " + (h.a[0] == h.a[0]), "false true");
        h.a[0].i = 5;
        check("fields", h.a[0].i + " " + h.a[1].i, "5 0");
        var m = [{i: 2}];
        check("element string", Std.string(m[0]) + " " + Std.string(h.a[0]), "{i: 2} {i: 5}");
        Sys.println("CONFORMANCE_OK");
    }
}
