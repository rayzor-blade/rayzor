@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c3_loop {
    static function take(r:Res):Int { return r.v; }
    static function main():Void {
        var a = new Res(3);
        var s = 0;
        for (i in 0...3) { s += take(a); }   // EXPECT ERROR: moved on iteration 2
        Sys.println(s);
    }
}
