@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
@:safety @:move
class Session {
    public var id:Int;
    public function new(id:Int) { this.id = id; }
    @:consume
    public function closeWith(@:borrow r:Res, o:Res):Int { return id + r.v + o.v; }
}
class c17_consume_borrowed_param {
    static function main():Void {
        var s = new Session(1);
        var keep = new Res(2);
        var give = new Res(3);
        var a = s.closeWith(keep, give);
        // EXPECT SILENT: a consumed receiver holds slot 0, so the declared
        // parameters shift by one and `keep` is still the @:borrow one.
        Sys.println(a + keep.v);
    }
}
