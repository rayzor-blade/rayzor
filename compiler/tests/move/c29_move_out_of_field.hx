@:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class Holder { public var r:Res; public function new(r:Res) { this.r = r; } }
class c29_move_out_of_field {
    static function take(r:Res):Int { return r.v; }
    static function main():Void {
        var h = new Holder(new Res(1));
        // EXPECT ERROR: a field has no binding to track, so the same value
        // taken twice cannot be told from taken once. Refused rather than
        // silently accepted.
        Sys.println(take(h.r) + take(h.r));
    }
}
