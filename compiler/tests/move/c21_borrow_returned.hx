@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c21_borrow_returned {
    static function keep(@:borrow r:Res):Res { return r; }   // EXPECT ERROR
    static function main():Void {
        var a = new Res(1);
        var b = keep(a);
        b.v = 99;
        Sys.println(a.v);
    }
}
