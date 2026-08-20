@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c12_capture_after_move {
    static function take(r:Res):Int { return r.v; }
    static function main():Void {
        var a = new Res(1);
        var n = take(a);
        var f = function():Int { return a.v; };   // EXPECT ERROR: captures a moved binding
        Sys.println(n + f());
    }
}
