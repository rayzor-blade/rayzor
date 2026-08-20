import pkg.Res;
import pkg.Util;

class Main {
    static function main():Void {
        var a = new Res(1);
        var x = Util.peek(a);
        var y = Util.peek(a);   // EXPECT SILENT: @:borrow declared in another module
        Sys.println(x + y);
    }
}
