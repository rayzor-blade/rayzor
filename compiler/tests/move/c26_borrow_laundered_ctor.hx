@:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class Wrap { public var r:Res; public function new(r:Res) { this.r = r; } }
class c26_borrow_laundered_ctor {
    // Wrapping a borrow in a constructor and handing the wrapper back is the
    // same escape as returning the borrow: EXPECT ERROR.
    static function viaCtor(@:borrow r:Res):Wrap { return new Wrap(r); }
    static function main():Void {
        var a = new Res(1);
        var w = viaCtor(a);
        w.r.v = 99;
        Sys.println(a.v);
    }
}
