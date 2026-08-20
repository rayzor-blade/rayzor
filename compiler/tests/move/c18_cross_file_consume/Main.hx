import pkg.Session;

class Main {
    static function main():Void {
        var s = new Session(1);
        var a = s.close();
        var b = s.decode();   // EXPECT ERROR: @:consume declared in another module
        Sys.println(a + b);
    }
}
