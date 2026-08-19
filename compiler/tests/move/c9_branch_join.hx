@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c9_branch_join {
    static function take(r:Res):Int { return r.v; }
    static function main():Void {
        var a = new Res(1);
        var s = 0;
        if (a.v > 0) { s += take(a); } else { s += 7; }
        Sys.println(s + a.v);   // EXPECT ERROR: moved on one of the two paths
    }
}
