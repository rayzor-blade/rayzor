import pkg.Res;

class Main {
    static function take(r:Res):Int { return 1; }
    static function main():Void {
        var a = new Res(1);
        var n = take(a);
        var m = take(a);          // EXPECT ERROR: a already moved, Res imported
        Sys.println(n + m);
    }
}
