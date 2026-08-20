@:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class Wrap { public var r:Res; public function new(r:Res) { this.r = r; } }
class c27_borrow_laundered_local {
    // Binding the wrapper first does not launder it: EXPECT ERROR.
    static function viaLocal(@:borrow r:Res):Wrap { var w = new Wrap(r); return w; }
    static function main():Void { Sys.println(viaLocal(new Res(1)).r.v); }
}
