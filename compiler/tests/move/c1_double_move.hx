@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c1_double_move {
    static function take(r:Res):Int { return r.v; }
    static function main():Void {
        var a = new Res(1);
        var n = take(a);
        var m = take(a);          // EXPECT ERROR: a already moved
        Sys.println(n + m);
    }
}
