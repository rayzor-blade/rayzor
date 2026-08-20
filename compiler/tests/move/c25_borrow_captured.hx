@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c25_borrow_captured {
    static function grab(@:borrow r:Res):Void {
        var f = function():Int { return r.v; };   // EXPECT ERROR: closure outlives the call
        Sys.println(f());
    }
    static function main():Void { grab(new Res(1)); }
}
