@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c22_borrow_returned_alias {
    static function keep(@:borrow r:Res):Res { var t = r; return t; }  // EXPECT ERROR via alias
    static function main():Void {
        var a = new Res(1);
        Sys.println(keep(a).v);
    }
}
