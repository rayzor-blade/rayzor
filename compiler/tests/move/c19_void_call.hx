@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c19_void_call {
    // Void on purpose: a void call lowers to no register, and an early return
    // on that once dropped every event for a call in statement position.
    static function take(r:Res):Void { Sys.println(r.v); }
    static function main():Void {
        var a = new Res(1);
        take(a);
        take(a);          // EXPECT ERROR: a already moved
    }
}
