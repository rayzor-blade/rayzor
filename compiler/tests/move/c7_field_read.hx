@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c7_field_read {
    static function take(r:Res):Int { return r.v; }
    static function main():Void {
        var a = new Res(1);
        var n = take(a);
        Sys.println(n + a.v);   // EXPECT ERROR: reading a field is still a use
    }
}
