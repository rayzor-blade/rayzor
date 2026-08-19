@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c2_return_branch {
    static function take(r:Res):Int { return r.v; }
    static function pick(r:Res, flag:Bool):Int {
        if (flag) return take(r);   // moved only on this path
        return r.v;                 // EXPECT SILENT: unreachable when moved
    }
    static function main():Void { Sys.println(pick(new Res(2), true)); }
}
