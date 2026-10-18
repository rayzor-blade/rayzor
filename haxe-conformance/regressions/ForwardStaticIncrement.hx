// A static field with a literal initializer keeps its type when another class,
// typed before it, increments or assigns it.
class Other {
    public static function post() { Main.a++; }
    public static function pre() { ++Main.b; }
    public static function plus() { Main.c += 1; }
    public static function rename() { Main.s += "!"; }
    public static function scale() { Main.f *= 2; }
}

class Main {
    public static var a = 0;
    public static var b = 0;
    public static var c = 0;
    public static var s = "x";
    public static var f = 1.5;

    static function main() {
        for (i in 0...3) {
            Other.post();
            Other.pre();
            Other.plus();
            Other.rename();
            Other.scale();
        }
        if (a != 3 || b != 3 || c != 3) throw "increments";
        if (s != "x!!!") throw "string";
        if (f != 12.0) throw "float";
        Sys.println("CONFORMANCE_OK");
    }
}
