@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c13_capture_before_move {
    static function take(r:Res):Int { return r.v; }
    static function main():Void {
        var a = new Res(1);
        var f = function():Int { return a.v; };   // EXPECT SILENT: captured while live
        var n = take(a);
        Sys.println(n + f());
    }
}
